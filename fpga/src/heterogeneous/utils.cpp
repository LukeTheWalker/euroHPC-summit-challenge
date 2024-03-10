//
// Created by tomma on 04/03/2024.
//

#include "utils.h"



void check_cl(cl_int err, const std::string& msg) {
    if(err != CL_SUCCESS) {
        std::cout << "error with code:  " << err << std::endl;
        std::cout << "message: " << msg << std::endl;
        exit(1);
    }
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

cl_mem allocateDeviceSingleInt(cl_int* err, cl_context context) {
    cl_mem ret = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL, err);
    return ret;
}


cl_mem allocateDeviceSingleDouble(cl_int* err, cl_context context) {
    cl_mem ret = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double), NULL, NULL, err);
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

void writeToBufferNoBlock(cl_command_queue queue, cl_mem buffer, size_t offset, size_t size, const double* host_buffer, size_t host_array_offset) {
    check_cl(clEnqueueWriteBuffer(queue, buffer, CL_FALSE, offset * sizeof(double ), size * sizeof(double), host_buffer + host_array_offset, 0, NULL, NULL), "error in writing to buffer");
}


void linkBufferToDevice(cl_command_queue queue, cl_mem buffer) {
    check_cl(clEnqueueMigrateMemObjects(queue, 1, &buffer, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, NULL),"error in migrating buffer");
}


void init_cl(cl_uint index_platform,  cl_command_queue** queues, cl_context* context, cl_device_id** mydev, cl_uint* number_device_found) {
    cl_int err;
    cl_uint number_platform_found;
    cl_platform_id* myp = (cl_platform_id*)malloc(MAX_PLATFORM*sizeof(cl_platform_id));
    check_cl(clGetPlatformIDs(MAX_PLATFORM, myp, &number_platform_found), "error in finding platforms");
    *mydev = (cl_device_id*)malloc(MAX_DEVICE * sizeof(cl_device_id));
    check_cl(clGetDeviceIDs(myp[index_platform], CL_DEVICE_TYPE_ACCELERATOR, MAX_DEVICE, *mydev, number_device_found), "error in finding devices");

    *context = clCreateContext(NULL, *number_device_found, *mydev, NULL, NULL, &err);
    check_cl(err, "error in creating the context");
    *queues = (cl_command_queue*)malloc(sizeof(cl_command_queue) * (*number_device_found));
    for(cl_uint i = 0; i < *number_device_found; i++) {
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
