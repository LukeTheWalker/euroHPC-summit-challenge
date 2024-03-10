//
// Created by tomma on 05/03/2024.
//



#include "utils.h"
#include <chrono>
#include <cmath>
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

void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    // y = alpha * A * x + beta * y;

    for(size_t r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}


cl_int init_cl(cl_uint device_numbers, cl_command_queue** queues, cl_context* context, cl_device_id** mydev) {
    cl_int err;
    cl_uint num_plat_found;
    cl_platform_id* myp = (cl_platform_id*)malloc(2*sizeof(cl_platform_id));

    err = clGetPlatformIDs(2, myp, &num_plat_found);
    /*for(int i = 0; i < num_plat_found; i++) {
        char name[100];
        clGetPlatformInfo(myp[i],CL_PLATFORM_NAME, 100, name, NULL);
        std::cout << name << std::endl;
    }*/

    std::cout << "found platform " << num_plat_found << std::endl;

    *mydev = (cl_device_id*)malloc(device_numbers * sizeof(cl_device_id));

    cl_uint found_device_n;
    err = clGetDeviceIDs(myp[1], CL_DEVICE_TYPE_ACCELERATOR, device_numbers, *mydev, &found_device_n);
    if(err == CL_DEVICE_NOT_FOUND) {
        std::cout << "no device found" << std::endl;
    }

    if(device_numbers > found_device_n) {
        std::cerr << "not enough devices : " << found_device_n << std::endl;
        free(*mydev);
        exit(1);
    }


    *context = clCreateContext(NULL, device_numbers, *mydev, NULL, NULL, &err);


    *queues = (cl_command_queue*)malloc(sizeof(cl_command_queue) * device_numbers);
    for(cl_uint i = 0; i < device_numbers; i++) {
        (*queues)[i] = clCreateCommandQueueWithProperties(*context, (*mydev)[i], 0, &err);
    }

    return err;
}


void generate_matrix(size_t n, double** matrix_out) {
    auto* matrix = new (std::align_val_t(MEM_ALIGNMENT)) double[n * n];
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
    auto* rhs = new (std::align_val_t(MEM_ALIGNMENT)) double[n];
    for(size_t i = 0; i < n; i++) {
        rhs[i] = value;
    }
    *rhs_out = rhs;
}


void conjugate_gradient(const double* matrix, const double* rhs, double* x, int size, double tol, int max_iters, cl_context context, cl_command_queue queue, cl_kernel kernel) {

    std::cout << "starting conjugate gradient" << std::endl;
    cl_mem device_matrix;
    cl_mem device_rhs;
    cl_mem sol;
    cl_mem p;
    cl_mem Ap;
    cl_mem r;
    cl_mem iter_number;
    cl_mem achieved_error;
    cl_int err = 0;



    device_matrix = allocateDeviceReadOnly(&err, size * size, context);
    check_cl(err, "error allocating matrix");
    device_rhs = allocateDeviceReadOnly(&err, size, context);
    check_cl(err, "error allocating rhs");
    sol = allocateDevice(&err, size, context);
    check_cl(err, "error allocating the solution");
    p = allocateDevice(&err, size, context);
    check_cl(err, "error allocating p");
    Ap = allocateDevice(&err, size, context);
    check_cl(err, "error allocating Ap");
    r = allocateDevice(&err, size, context);
    check_cl(err, "error allocating r");
    iter_number = allocateDeviceSingleInt(&err, context);
    check_cl(err, "error allocating iter_number");
    achieved_error = allocateDeviceSingleDouble(&err, context);
    check_cl(err, "error allocating achieved error");

    writeToBufferNoBlock(queue, p, 0, size, rhs, 0);
    writeToBufferNoBlock(queue, device_matrix, 0, size * size, matrix, 0);
    writeToBufferNoBlock(queue, device_rhs, 0, size, rhs, 0);
    writeToBufferNoBlock(queue, sol, 0, size, x, 0);
    writeToBufferNoBlock(queue, r, 0, size, rhs, 0);

    double squared_tol = tol * tol;
    double bb = dot(rhs, rhs, size);

    clFinish(queue);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_matrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_rhs);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &sol);
    clSetKernelArg(kernel, 3, sizeof(int), &size);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &r);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &p);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &Ap);
    clSetKernelArg(kernel, 7, sizeof(int), &max_iters);
    clSetKernelArg(kernel, 8, sizeof(double), &squared_tol);
    clSetKernelArg(kernel, 9, sizeof(double), &bb);
    clSetKernelArg(kernel, 10, sizeof(cl_mem), &iter_number);
    clSetKernelArg(kernel, 11, sizeof(cl_mem), &achieved_error);

    cl_event wait_finish_kernel;
    clEnqueueTask(queue, kernel, 0, NULL, &wait_finish_kernel);
    clEnqueueReadBuffer(queue, sol, CL_TRUE, 0, size * sizeof(double), x, 1, &wait_finish_kernel, NULL);

    double* tmp = new double[size];
    for(int i = 0; i < size; i++) {
        tmp[i] = 0.0;
        for(int j = 0; j < size; j++) {
            tmp[i] += matrix[i*size+j]*x[j];
        }
    }
    double res_err = 0;
    for(int i = 0; i < size; i++) {
        //std::cout << x[i] << " " << rhs[i] << " " << tmp[i] << std::endl;
        res_err += (rhs[i] - tmp[i])*(rhs[i] - tmp[i]);
    }
    std::cout << "error: " << res_err << std::endl;

}

void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
{
    double alpha, beta, bb, rr, rr_new;
    double * r = new double[size];
    double * p = new double[size];
    double * Ap = new double[size];
    int num_iters;

    for(size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(1.0, A, p, 0.0, Ap, size, size);
        alpha = rr / dot(p, Ap, size);
        axpby(alpha, p, 1.0, x, size);
        axpby(-alpha, Ap, 1.0, r, size);
        rr_new = dot(r, r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
        axpby(1.0, r, beta, p, size);
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}



int main(int argc, char** argv) {
    int max_iters = atoi(argv[3]);
    double tol = atof(argv[4]);
    cl_context context;
    cl_command_queue* command_queues;
    cl_program program;
    cl_device_id* devices;
    if(init_cl(2, &command_queues, &context, &devices)!=0) {
        std::cout << "error" << std::endl;
    }
    load_program("../src/fpga/CG_improved_v1.aocx", &program, context, 1, devices);

    cl_int err;
    cl_kernel kernel  = create_kernel(program, "conjugate_gradient_kernel", &err);
    size_t size;
    double* rhs;
    double* matrix;
    read_matrix_from_file(argv[1], &matrix, &size, &size);
    double* sol = new (std::align_val_t(MEM_ALIGNMENT)) double[size];

    for(int i = 0; i < size; i++) {
        sol[i] = 0.0;
    }
    size_t tmp;
    read_matrix_from_file(argv[2], &rhs, &tmp, &tmp);




    auto start_fpga = std::chrono::high_resolution_clock::now();
    conjugate_gradient(matrix, rhs, sol, size, tol, max_iters, context, command_queues[0], kernel);
    auto stop_fpga = std::chrono::high_resolution_clock::now();

    auto start = std::chrono::high_resolution_clock::now();
    conjugate_gradients(matrix, rhs, sol, size, max_iters, tol);
    auto stop = std::chrono::high_resolution_clock::now();

    long execution_time_fpga = std::chrono::duration_cast<std::chrono::microseconds>(stop_fpga - start_fpga).count();
    long execution_time_cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    std::cout << "fpga: " << execution_time_fpga << std::endl;
    std::cout << "cpu: " << execution_time_cpu << std::endl;



}