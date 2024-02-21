#ifndef MULTI_GPU_LUCA_HPP
#define MULTI_GPU_LUCA_HPP

#include <cstdio>
#include <cuda_runtime.h>
#include <utils.cuh>
#include <conjugate_gradients_gpu.cu>
#include <nccl.h>

#define nranks 4

namespace luca {

ncclComm_t comms[nranks];
int devs[nranks] = { 0, 1, 2, 3 };


__global__ void transpose_matrix (const double * A, double * At, size_t num_rows, size_t num_cols)
{
    // use shared memory to store the tile
    __shared__ double tile[32][32];

    // calculate the row and column index
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // load the tile into shared memory
    if (row < num_rows && col < num_cols)
    {
        tile[threadIdx.y][threadIdx.x] = A[row * num_cols + col];
    }

    // synchronize threads
    __syncthreads();

    // calculate the new row and column index
    row = blockIdx.x * blockDim.x + threadIdx.y;
    col = blockIdx.y * blockDim.y + threadIdx.x;

    // store
    if (row < num_cols && col < num_rows)
    {
        At[row * num_rows + col] = tile[threadIdx.x][threadIdx.y];
    }
}

void gemv_mutli_gpu_tiled_kernel_launcher(const double ** local_A, const double * x, double * y, size_t * num_rows_per_device, size_t num_cols, cudaStream_t * s)
{
    int number_of_devices; cudaError_t err; ncclResult_t nccl_err;

    err = cudaGetDeviceCount(&number_of_devices); cuda_err_check(err, __FILE__, __LINE__);

    int threadsPerRow = 10;
    size_t sharedMemSize = num_cols / threadsPerRow * sizeof(double);

    double ** y_partial_local = (double**)malloc(number_of_devices * sizeof(double*));
    double ** y_local = (double**)malloc(number_of_devices * sizeof(double*));

    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);

        int rowsperblock = 1024;
        // Define the size of the grid and blocks
        dim3 blockDim(1, rowsperblock);
        dim3 gridDim(threadsPerRow, (num_rows_per_device[i] + rowsperblock - 1) / rowsperblock);

        err = cudaMalloc((void**)&y_partial_local[i], num_rows_per_device[i] * threadsPerRow * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc((void**)&y_local[i], num_rows_per_device[i] * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemset(y_partial_local[i], 0, num_rows_per_device[i] * threadsPerRow * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

        // Launch the kernel
        gemv_tiled_kernel<<<gridDim, blockDim, sharedMemSize, s[i]>>>(local_A[i], x, y_partial_local[i], num_rows_per_device[i], num_cols);
        reduce_rows<<<(num_rows_per_device[i] + threadsPerRow - 1) / threadsPerRow, threadsPerRow>>>(y_partial_local[i], y_local[i], num_rows_per_device[i], threadsPerRow);

    }

    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamSynchronize(s[i]); cuda_err_check(err, __FILE__, __LINE__);
    }

    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpyPeerAsync(y + i * (num_rows_per_device[i]), i, y_local[i], i, num_rows_per_device[i] * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(y_partial_local[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(y_local[i]); cuda_err_check(err, __FILE__, __LINE__);
    }

    // sync all streams
    for(int i = 0; i < number_of_devices; i++) err = cudaStreamSynchronize(s[i]); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaSetDevice(0); cuda_err_check(err, __FILE__, __LINE__);

    free(y_partial_local);
    free(y_local);
}



void par_conjugate_gradients_multi_gpu(const double * h_A, const double * h_b, double * h_x, size_t size, int max_iters, double rel_error)
{
    cudaError_t err; ncclResult_t nccl_err;

    nccl_err = ncclCommInitAll(comms, nranks, devs); nccl_err_check(nccl_err, __FILE__, __LINE__);

    const double /* d_A,*/ * d_b;
    int num_iters;

    double alpha, beta, bb, rr, rr_new;
    double * d_r, * d_p, * d_Ap, * d_x;

    int number_of_devices;
    cudaStream_t * s;
    size_t * number_of_rows_per_device;
    const double ** d_local_A, ** d_local_A_transposed;

    err = cudaGetDeviceCount(&number_of_devices); cuda_err_check(err, __FILE__, __LINE__);
    s = (cudaStream_t*)malloc(number_of_devices * sizeof(cudaStream_t));
    d_local_A = (const double**)malloc(number_of_devices * sizeof(const double*));
    d_local_A_transposed = (double**)malloc(number_of_devices * sizeof(double*));
    number_of_rows_per_device = (size_t*)malloc(number_of_devices * sizeof(size_t));

    for(int i = 0; i < number_of_devices; i++)
    {   
        number_of_rows_per_device[i] = (i == number_of_devices - 1) ? size - i * (size / number_of_devices) : size / number_of_devices;
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&d_local_A[i], size * number_of_rows_per_device[i] * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&d_local_A_transposed[i], size * number_of_rows_per_device[i] * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpyAsync((void*)d_local_A[i], h_A + i * (size / number_of_devices) * size, size * number_of_rows_per_device[i] * sizeof(double), cudaMemcpyHostToDevice, s[i]); cuda_err_check(err, __FILE__, __LINE__);
        transpose_matrix<<<(size + 31) / 32, dim3(32, 32), 0, s[i]>>>(d_local_A[i], (double*)d_local_A_transposed[i], size, number_of_rows_per_device[i]); 
        err = cudaFreeAsync((void*)d_local_A[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
    }

    // sync all streams
    for(int i = 0; i < number_of_devices; i++) err = cudaStreamSynchronize(s[i]); cuda_err_check(err, __FILE__, __LINE__);
    
    err = cudaSetDevice(0); cuda_err_check(err, __FILE__, __LINE__);

    // err = cudaMalloc((void**)&d_A, size * size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_b, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMalloc((void**)&d_r, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_p, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_Ap, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_x, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    // err = cudaMemcpy((void*)d_A, h_A, size * size * sizeof(double), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy((void*)d_b, h_b, size * sizeof(double), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemset(d_x, 0, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_r, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_p, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

    bb = dot_kernel_launcher(d_b, d_b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // gemv(1.0, A, p, 0.0, Ap, size, size);
        gemv_mutli_gpu_tiled_kernel_launcher(d_local_A, d_p, d_Ap, number_of_rows_per_device, size, s);
        // gemv_kernel_launcher(1.0, d_A, d_p, 0.0, d_Ap, size, size);
        // alpha = rr / dot(p, Ap, size);
        alpha = rr / dot_kernel_launcher(d_p, d_Ap, size);
        // axpby(alpha, p, 1.0, x, size);
        axpby_kernel_launcher(alpha, d_p, 1.0, d_x, size);
        // axpby(-alpha, Ap, 1.0, r, size);
        axpby_kernel_launcher(-alpha, d_Ap, 1.0, d_r, size);
        // rr_new = dot(r, r, size);
        rr_new = dot_kernel_launcher(d_r, d_r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
        // axpby(1.0, r, beta, p, size);
        axpby_kernel_launcher(1.0, d_r, beta, d_p, size);
    }

    transfer_to_host(d_x, h_x, size);

    // err = cudaFree((void*)d_A); cuda_err_check(err, __FILE__, __LINE__);
    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree((void*)d_local_A[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamDestroy(s[i]); cuda_err_check(err, __FILE__, __LINE__);
    }
    err = cudaFree(s); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree((void*)d_b); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_r); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_p); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_Ap); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_x); cuda_err_check(err, __FILE__, __LINE__);

    for (int i=0; i<nranks; i++)
        ncclCommDestroy(comms[i]);

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}
}

#endif