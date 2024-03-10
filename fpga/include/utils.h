//
// Created by tomma on 02/03/2024.
//

#ifndef MATRIX_VECTOR_MULTIPLICATION_UTILS_H
#define MATRIX_VECTOR_MULTIPLICATION_UTILS_H

#include "../CL/cl.h"
#include <iostream>
#include <vector>
#include <fstream>
#define MAX_PLATFORM 10
#define MAX_DEVICE 10
#define MATRIX_VECTOR_KERNEL_PATH "../src/fpga/MVP_improved_v1.aocx"
#define MATRIX_VECTOR_KERNEL_NAME "matrix_vector_kernel"
#define CL_DEVICE_GLOBAL_MEM_SIZE 1024*1024*1024*16

struct MatrixData {
    size_t offset;
    size_t partial_size;
};

template<typename Vector1, typename Vector2>
inline double dot(const Vector1& x, const Vector2& y, size_t size)
{
    double result = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}


template<typename Vector1, typename Vector2>
inline void axpby(double alpha, const Vector1& x, double beta, Vector2& y, size_t size)
{
    // y = alpha * x + beta * y

    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void check_cl(cl_int err, const std::string& msg);

cl_mem allocateDeviceSingleInt(cl_int* err, cl_context context);


cl_mem allocateDeviceSingleDouble(cl_int* err, cl_context context);


void load_program(const std::string& path, cl_program* program, cl_context context, cl_uint num_devices, const cl_device_id* device_list);

cl_kernel create_kernel(cl_program program, const char* kernel_name, cl_int* errorcode);

cl_mem allocateDeviceReadOnly(const double* host_array, cl_int* err, size_t size, int offset, cl_context context, cl_command_queue queue);

cl_mem allocateDevice(const double* host_array, cl_int* err, size_t size, int offset, cl_context context, cl_command_queue queue);
cl_mem allocateDevice(cl_int* err, size_t size, cl_context context);

cl_mem allocateDeviceReadOnly(cl_int* err, size_t size, cl_context context);

void writeToBuffer(cl_command_queue queue, cl_mem buffer, size_t offset, size_t size, const double* host_buffer, size_t host_array_offset);

void writeToBufferNoBlock(cl_command_queue queue, cl_mem buffer, size_t offset, size_t size, const double* host_buffer, size_t host_array_offset);


void linkBufferToDevice(cl_command_queue queue, cl_mem buffer);


void init_cl(cl_uint index_platform,  cl_command_queue** queues, cl_context* context, cl_device_id** mydev, cl_uint* number_device_found);

void matrix_vector_multiplication(double* host_Ap, size_t offset, cl_mem* A, cl_mem* p, cl_mem* Ap, size_t nrows, size_t ncols, cl_command_queue* queue, cl_kernel* matrix_vector_kernel);

#endif //MATRIX_VECTOR_MULTIPLICATION_UTILS_H
