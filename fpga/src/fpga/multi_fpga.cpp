//
// Created by tomma on 29/02/2024.
//

#include "CL/opencl.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <math.h>
#include <omp.h>
#include <chrono>

#define MAX_PLATFORM 10
#define MATRIX_VECTOR_KERNEL_PATH "../src/fpga/MVV.aocx"
#define MATRIX_VECTOR_KERNEL_NAME "matrix_vector_kernel"
#define MEM_ALIGNMENT 64


bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}

double dot(const double * x, const double * y, size_t size)
{
    double result = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}



void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    // y = alpha * x + beta * y

    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void gemv(const double * A, const double * x,double * y, size_t num_rows, size_t num_cols)
{

    for(size_t r = 0; r < num_rows; r++)
    {
        y[r] = 0.0;
        for(size_t c = 0; c < num_cols; c++)
        {
            y[r] += A[r * num_cols + c] * x[c];
        }

    }
}


void check_cl(cl_int err, const std::string& msg) {
    if(err != CL_SUCCESS) {
        std::cout << "error with code:  " << err << std::endl;
        std::cout << "message: " << msg << std::endl;
        exit(1);
    }
}

void generate_matrix(size_t n, double** matrix_out) {
    auto* matrix = new double[n * n];
    for(size_t i = 0; i < n * n; i++) {
        matrix[i] = 0.0;
    }
    for(size_t i = 0; i < n; i++) {
        matrix[i*n + i] = 2.0;
        if(i != n-1) {
            matrix[(i+1)*n + i] = -1;
            matrix[i*n + (i+1)] = -1;
        }
    }
    *matrix_out = matrix;
}

void generate_rhs(size_t n, double value, double** rhs_out) {
    auto* rhs = new double[n];
    for(size_t i = 0; i < n; i++) {
        rhs[i] = value;
    }
    *rhs_out = rhs;
}

cl_mem allocateDeviceReadOnly(const double* host_array, cl_int* err, size_t size, int offset, cl_context context, cl_command_queue queue) {
    cl_mem ret = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(double), NULL, NULL, err);
    check_cl(*err, "Error in creating Read-Only buffer");
    *err = clEnqueueWriteBuffer(queue, ret, CL_TRUE, offset * sizeof(double), size * sizeof(double), host_array, 0, NULL, NULL);
    check_cl(*err, "Error in writing to Read-Only buffer");
    return ret;
}

cl_mem allocateDevice(const double* host_array, cl_int* err, size_t size, int offset, cl_context context, cl_command_queue queue) {
    cl_mem ret = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, NULL, err);
    check_cl(*err, "Error in creating Read-Write buffer");
    *err = clEnqueueWriteBuffer(queue, ret, CL_TRUE, offset * sizeof(double), size * sizeof(double), host_array, 0, NULL, NULL);
    check_cl(*err, "Error in writing Read-Write buffer");
    return ret;
}

cl_mem allocateDevice(cl_int* err, size_t size, cl_context context) {
    cl_mem ret = clCreateBuffer(context, CL_MEM_READ_WRITE, size * sizeof(double), NULL, NULL, err);
    check_cl(*err, "Error in creating uninitialized buffer");
    return ret;
}

cl_mem allocateDeviceReadOnly(cl_int* err, size_t size, cl_context context) {
    cl_mem ret = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(double), NULL, NULL, err);
    check_cl(*err, "Error in creating uninitialized Read-Only buffer");
    return ret;
}

void writeToBuffer(cl_command_queue queue, cl_mem buffer, size_t offset, size_t size, const double* host_buffer, size_t host_array_offset) {
    check_cl(clEnqueueWriteBuffer(queue, buffer, CL_TRUE, offset * sizeof(double ), size * sizeof(double), host_buffer + host_array_offset, 0, NULL, NULL), "error in writing to buffer");
}

void linkBufferToDevice(cl_command_queue queue, cl_mem buffer) {
    check_cl(clEnqueueMigrateMemObjects(queue, 1, &buffer, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, NULL),"error in migrating buffer");
}


void init_cl(cl_uint index_platform, cl_uint device_numbers, cl_command_queue** queues, cl_context* context, cl_device_id** mydev) {
    cl_int err;
    cl_uint number_platform_found;
    cl_platform_id* myp = (cl_platform_id*)malloc(MAX_PLATFORM*sizeof(cl_platform_id));
    check_cl(clGetPlatformIDs(MAX_PLATFORM, myp, &number_platform_found), "error in finding platforms");
    *mydev = (cl_device_id*)malloc(device_numbers * sizeof(cl_device_id));
    cl_uint number_device_found;
    check_cl(clGetDeviceIDs(myp[index_platform], CL_DEVICE_TYPE_ACCELERATOR, device_numbers, *mydev, &number_device_found), "error in finding devices");
    if(device_numbers > number_device_found) {
        std::cerr << "not enough devices : " << number_device_found << std::endl;
        exit(1);
    }


    *context = clCreateContext(NULL, device_numbers, *mydev, NULL, NULL, &err);
    check_cl(err, "error in creating the context");
    *queues = (cl_command_queue*)malloc(sizeof(cl_command_queue) * device_numbers);
    for(cl_uint i = 0; i < device_numbers; i++) {
        (*queues)[i] = clCreateCommandQueueWithProperties(*context, (*mydev)[i], 0, &err);
        check_cl(err, "error in creating the command queue for device " + std::to_string(i));
    }

    free(myp);
    std::cout << "Successfully initialized OpenCL FPGA Environment" << std::endl;
}

void matrix_vector_multiplication(double* host_Ap, size_t offset, cl_mem* A, cl_mem* p, cl_mem* Ap, size_t nrows, size_t ncols, cl_command_queue* queue, cl_kernel* matrix_vector_kernel) {
    check_cl(clSetKernelArg(*matrix_vector_kernel, 0, sizeof(cl_mem), A), "error setting first kernel argument");
    check_cl(clSetKernelArg(*matrix_vector_kernel, 1, sizeof(cl_mem), p), "error setting second kernel argument");
    check_cl(clSetKernelArg(*matrix_vector_kernel, 2, sizeof(cl_mem), Ap), "error setting third kernel argument");
    check_cl(clSetKernelArg(*matrix_vector_kernel, 3, sizeof(int), &nrows), "error setting fourth kernel argument");
    check_cl(clSetKernelArg(*matrix_vector_kernel, 4, sizeof(int), &ncols), "error setting fifth kernel argument");

    cl_event wait_finish_kernel;
    check_cl(clEnqueueTask(*queue, *matrix_vector_kernel, 0, NULL, &wait_finish_kernel), "error launching the kernel");
    check_cl(clEnqueueReadBuffer(*queue, *Ap, CL_TRUE, 0, nrows*sizeof(double), host_Ap + offset, 0, NULL, NULL), "error with reading back the solution");
}

void check_product(const double* array1, const double* array2, size_t size) {
    double err = 0;
    for(int i = 0; i < size; i++) {
        err += (array1[i] - array2[i]) * (array1[i] - array2[i]);
        //std::cout << array1[i] << " "  << array2[i] << std::endl;
    }

    /*if(err > 1e-12) {
        std::cout << "error in matrix_vector multiplication: " << err << std::endl;
        exit(1);
    }*/
}

void split_matrix(const double* matrix, size_t split_number, double** splitted_matrix, size_t* offset, size_t* partial_size, size_t size) {

    for(int i = 0; i < split_number; i++) {
        splitted_matrix[i] = new double[partial_size[i] * size];
        for(int j = 0; j < partial_size[i] * size; j++) {
            splitted_matrix[i][j] = matrix[j + offset[i] * size];
        }
    }
}


void conjugate_gradient_aligned(const double* A, const double* b, double* x, size_t size, int max_iters, double tol, int device_number, cl_command_queue* queues, cl_context context, cl_kernel* kernels) {
    std::cout << "starting conjugate gradient" << std::endl;
    double alpha;
    double beta;
    double rr;
    double rr_new;
    double bb;
    cl_int err;
    int iters;
    size_t* offset = new size_t[device_number];
    size_t* partial_size = new size_t[device_number];
    double** splitted_matrix = new double* [device_number];
    double** splitted_Ap = new double* [device_number];
    split_matrix(A, device_number, splitted_matrix, offset, partial_size, size);
    for(int i = 0; i < device_number; i++) {
        splitted_Ap[i] = new double[partial_size[i]];
    }

    cl_mem* device_A = new cl_mem[device_number];
    cl_mem* device_p = new cl_mem[device_number];
    cl_mem* device_Ap = new cl_mem[device_number];

    double* p = new double[size];
    double* r = new double[size];

    for(int i = 0; i < size; i++) {
        p[i] = b[i];
        r[i] = b[i];
        x[i] = 0.0;
    }


    bb = dot(b,b,size);
    rr = bb;
    for(int i = 0; i < device_number; i++) {

        device_A[i] = allocateDeviceReadOnly(&err, partial_size[i] * size, context);
        linkBufferToDevice(queues[i], device_A[i]);
        writeToBuffer(queues[i], device_A[i], 0, partial_size[i] * size, splitted_matrix[i], 0);

        device_p[i] = allocateDevice(&err, size, context);
        linkBufferToDevice(queues[i], device_p[i]);


        device_Ap[i] = allocateDevice(&err, partial_size[i], context);
        linkBufferToDevice(queues[i], device_Ap[i]);
    }

    for(iters = 1; iters <= max_iters; iters++) {
        double tmp = 0;
        for(int i = 0; i < device_number; i++) {
            writeToBuffer(queues[i], device_p[i], 0, size, p, 0);
            matrix_vector_multiplication(splitted_Ap[i], 0, &(device_A[i]), &(device_p[i]), &(device_Ap[i]), partial_size[i], size, &(queues[i]), &(kernels[i]));
        }


        //alpha = rr / dot(p, Ap, size);
        int current_loop_device = 0;
        int current_loop_device_it = 0;
        for(int i = 0; i < size; i++, current_loop_device_it++) {
            if(offset[current_loop_device + 1] == i) {
                current_loop_device++;
                current_loop_device_it = 0;
            }
            tmp += p[i] * splitted_Ap[current_loop_device][current_loop_device_it];
        }
        axpby(alpha, p, 1.0, x, size);
        //axpby(-alpha, Ap, 1.0, r, size);
        current_loop_device_it = 0;
        current_loop_device = 0;
        for(int i = 0; i < size; i++) {
            if(offset[current_loop_device + 1] == i) {
                current_loop_device++;
                current_loop_device_it = 0;
            }
            r[i] += -alpha * splitted_Ap[current_loop_device][current_loop_device_it];
        }
        rr_new = dot(r, r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < tol) { break; }
        axpby(1.0, r, beta, p, size);
    }

    if(iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
    std::cout << "finished conjugate gradient" << std::endl;
}

void conjugate_gradient_aligned2(const double* A, const double* b, double* x, size_t size, int max_iters, double tol, int device_number, cl_command_queue* queues, cl_context context, cl_kernel* kernels) {
    double alpha, beta, rr, rr_new, bb;
    cl_int err = 0;
    int iters;
    size_t* offset = new size_t[device_number];
    size_t* partial_size = new size_t[device_number];
    double** splitted_A = new double * [device_number];
    double** splitted_Ap = new double * [device_number];

    offset[0] = 0.0;
    for(int i = 1; i < device_number; i++) {
        offset[i] = size/device_number + offset[i-1];
    }

    for(int i = 0; i < device_number; i++) {
        if(i != device_number - 1) {
            partial_size[i] = offset[i+1] - offset[i];
        } else {
            partial_size[device_number - 1] = size - offset[device_number - 1];
        }
    }

    for(int i = 0; i < device_number; i++) {
        splitted_A[i] = new (std::align_val_t(MEM_ALIGNMENT))double  [partial_size[i] * size];
        splitted_Ap[i] = new (std::align_val_t(MEM_ALIGNMENT))double [partial_size[i]];
        for(int j = 0; j < size * partial_size[i]; j++) {
            splitted_A[i][j] = A[size*offset[i] + j];
        }
    }

    cl_mem* device_A = new cl_mem[device_number];
    cl_mem* device_p = new cl_mem[device_number];
    cl_mem* device_Ap = new cl_mem[device_number];

    double* p = new(std::align_val_t(MEM_ALIGNMENT)) double[size];
    double* r = new double[size];

    for(int i = 0; i < size; i++) {
        p[i] = b[i];
        r[i] = b[i];
        x[i] = 0.0;
    }


    bb = dot(b,b,size);
    rr = bb;

    for(int i = 0; i < device_number; i++) {
        device_A[i] = allocateDeviceReadOnly(&err, partial_size[i] * size, context);
        linkBufferToDevice(queues[i], device_A[i]);
        writeToBuffer(queues[i], device_A[i], 0, partial_size[i] * size, splitted_A[i], 0);
        device_p[i] = allocateDevice(&err, size, context);
        linkBufferToDevice(queues[i], device_p[i]);
        device_Ap[i] = allocateDevice(&err, partial_size[i], context);
        linkBufferToDevice(queues[i], device_Ap[i]);
    }
    int actual_iters;

#pragma omp parallel num_threads(device_number) default(none) firstprivate(iters) shared(actual_iters, max_iters, device_number, alpha, beta, rr_new, r, rr, bb, offset,x, tol) shared(queues, device_p, size, p, splitted_Ap, device_A, device_Ap, kernels, partial_size)
    {
        for (iters = 1; iters <= max_iters; iters++) {
#pragma omp for
            for (int i = 0; i < device_number; i++) {
                writeToBuffer(queues[i], device_p[i], 0, size, p, 0);
                matrix_vector_multiplication(splitted_Ap[i], 0, &(device_A[i]), &(device_p[i]), &(device_Ap[i]),
                                             partial_size[i], size, &(queues[i]), &(kernels[i]));
            }
#pragma omp single
            {
                actual_iters = iters;
                double tmp = 0.0;
                int current_loop_device = 0;
                int current_loop_device_it = 0;
                for (int i = 0; i < size; i++, current_loop_device_it++) {
                    if (current_loop_device + 1 != device_number && offset[current_loop_device + 1] == i) {
                        current_loop_device++;
                        current_loop_device_it = 0;
                    }
                    tmp += p[i] * splitted_Ap[current_loop_device][current_loop_device_it];
                }
                alpha = rr / tmp;
                axpby(alpha, p, 1.0, x, size);
                current_loop_device_it = 0;
                current_loop_device = 0;
                for (int i = 0; i < size; i++, current_loop_device_it++) {
                    if (current_loop_device + 1 != device_number && offset[current_loop_device + 1] == i) {
                        current_loop_device++;
                        current_loop_device_it = 0;
                    }
                    r[i] += -alpha * splitted_Ap[current_loop_device][current_loop_device_it];
                }
                rr_new = dot(r, r, size);
                beta = rr_new / rr;
                rr = rr_new;
                axpby(1.0, r, beta, p, size);
            }
            if (std::sqrt(rr / bb) < tol) { break; }
        }
    }

    if(actual_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", actual_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}

void conjugate_gradient(const double* A, const double* b, double* x, size_t size, int max_iters, double tol, int device_number, cl_command_queue* queues, cl_context context, cl_kernel* kernels) {
    std::cout << "starting conjugate gradient" << std::endl;
    double alpha;
    double beta;
    double rr;
    double rr_new;
    double bb;
    cl_int err;
    int iters;
    size_t* offset = new size_t[device_number];
    size_t* partial_size = new size_t[device_number];
    offset[0] = 0;
    for(int i = 1; i < device_number; i++) {
        offset[i] = offset[i-1] + size/device_number;
    }

    for(int i = 0; i < device_number; i++) {
        if(i != device_number - 1) {
            partial_size[i] = offset[i+1] - offset[i];
        } else {
            partial_size[device_number - 1] = size - offset[i];
        }
    }

    cl_mem* device_A = new cl_mem[device_number];
    cl_mem* device_p = new cl_mem[device_number];
    cl_mem* device_Ap = new cl_mem[device_number];

    double* Ap = new double[size];
    double* Ap_test = new double[size];
    double* p = new double[size];
    double* r = new double[size];

    for(int i = 0; i < size; i++) {
        p[i] = b[i];
        r[i] = b[i];
        x[i] = 0.0;
    }


    bb = dot(b,b,size);
    rr = bb;
    for(int i = 0; i < device_number; i++) {
        device_A[i] = allocateDeviceReadOnly(&err, partial_size[i] * size, context);
        linkBufferToDevice(queues[i], device_A[i]);
        writeToBuffer(queues[i], device_A[i], 0, partial_size[i] * size, A, offset[i] * size);

        device_p[i] = allocateDevice(&err, size, context);
        linkBufferToDevice(queues[i], device_p[i]);


        device_Ap[i] = allocateDevice(&err, partial_size[i], context);
        linkBufferToDevice(queues[i], device_Ap[i]);
    }

    for(iters = 1; iters <= max_iters; iters++) {
        for(int i = 0; i < device_number; i++) {
            writeToBuffer(queues[i], device_p[i], 0, size, p, 0);
            matrix_vector_multiplication(Ap, offset[i], &(device_A[i]), &(device_p[i]), &(device_Ap[i]), partial_size[i], size, &(queues[i]), &(kernels[i]));
        }

        gemv(A, p, Ap_test, size, size);
        check_product(Ap_test, Ap, size);
        alpha = rr / dot(p, Ap, size);
        axpby(alpha, p, 1.0, x, size);
        axpby(-alpha, Ap, 1.0, r, size);
        rr_new = dot(r, r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < tol) { break; }
        axpby(1.0, r, beta, p, size);
    }

    if(iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
    std::cout << "finished conjugate gradient" << std::endl;
}

void load_program(const std::string& path, cl_program* program, cl_context context, cl_uint num_devices, const cl_device_id* device_list) {
    std::ifstream input_file;
    input_file.open(path);
    if(!input_file.is_open()) {
        std::cout << "error in opening the file" << std::endl;
        exit(1);
    }
    size_t* length = new size_t[num_devices];
    unsigned char* buffer;
    input_file.seekg (0, std::ios::end);
    for(int i = 0; i < num_devices; i++) {
        length[i] = input_file.tellg();
    }
    input_file.seekg (0, std::ios::beg);
    buffer = new unsigned char [length[0]];
    input_file.read (reinterpret_cast<char *>(buffer), length[0]);
    input_file.close();
    const unsigned char** binaries = (const unsigned char**)malloc(sizeof(unsigned char*) * num_devices);
    for(int i = 0; i < num_devices; i++) {
        binaries[i] = buffer;
    }
    cl_int binary_status;
    cl_int errorcode_ret;

    *program = clCreateProgramWithBinary(context, num_devices, device_list, length, binaries, &binary_status, &errorcode_ret);
    check_cl(errorcode_ret, "error in building the program");
    free(binaries);
}

cl_kernel create_kernel(cl_program program, const char* kernel_name, cl_int* errorcode) {
    cl_kernel kernel = clCreateKernel(program, kernel_name, errorcode);
    check_cl(*errorcode, "error in creating the kernel");
    return kernel;
}



int main(int argc, char** argv) {
    size_t size;
    size_t tmp;
    int max_iters = atoi(argv[3]);
    double tol = atof(argv[4]);
    cl_int err = 0;
    int number_device_required = 2;
    int platform_index = 1;

    cl_command_queue* queues;
    cl_context context;
    cl_device_id* devices;
    cl_program program;
    cl_kernel* kernels = new cl_kernel[number_device_required];
    init_cl(platform_index, number_device_required, &queues, &context, &devices);
    load_program(MATRIX_VECTOR_KERNEL_PATH, &program, context, number_device_required, devices);

    for(int i = 0; i < number_device_required; i++) {
        kernels[i] = create_kernel(program, MATRIX_VECTOR_KERNEL_NAME, &err);
    }
    double* matrix;
    double* rhs;

    read_matrix_from_file(argv[1], &matrix, &size, &size);
    double* sol = new double[size];


    read_matrix_from_file(argv[2], &rhs, &tmp, &tmp);
    auto start = std::chrono::high_resolution_clock::now();
    conjugate_gradient_aligned2(matrix, rhs, sol, size, max_iters, tol, number_device_required, queues, context, kernels);
    auto stop = std::chrono::high_resolution_clock::now();
    long exe_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << exe_time << std::endl;

}
