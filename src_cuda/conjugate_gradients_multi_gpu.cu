#ifndef MULTI_GPU_LUCA_HPP
#define MULTI_GPU_LUCA_HPP

#include <cstdio>
#include <utils.cuh>
#include <conjugate_gradients_gpu.cu>
#include <chrono>
#include <omp.h>

#define nranks 4
#define TILE_DIM 32
#define BLOCK_ROWS 8

#define SHMEM 800

namespace luca {


__global__ void transpose(double *odata, const double *idata, size_t nrows, size_t ncols)
{
    __shared__ double block[TILE_DIM][TILE_DIM+1];
    size_t x = (size_t)blockIdx.x * TILE_DIM + (size_t)threadIdx.x;
    size_t y = (size_t)blockIdx.y * TILE_DIM + (size_t)threadIdx.y;
    if((x < ncols) && (y < nrows))
    {
        block[threadIdx.y][threadIdx.x] = idata[y*ncols + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if((x < nrows) && (y < ncols))
    {
        odata[y*nrows + x] = block[threadIdx.x][threadIdx.y];
    }       
}

void gemv_multi_gpu_tiled_kernel_launcher(const double ** local_A, const double * x, double * y, double ** y_partial_local, double ** y_local, double ** x_local, size_t sharedMemSize, size_t threadsPerRow, size_t * num_rows_per_device, size_t num_cols, cudaStream_t * s)
{
    int number_of_devices; cudaError_t err;

    err = cudaGetDeviceCount(&number_of_devices); cuda_err_check(err, __FILE__, __LINE__);

    // size_t threadsPerRow = 10;
    // size_t sharedMemSize = num_cols / threadsPerRow * sizeof(double);

    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);

        size_t rowsperblock = 1024;
        // Define the size of the grid and blocks
        dim3 blockDim(1, rowsperblock);
        dim3 gridDim(threadsPerRow, (num_rows_per_device[i] + rowsperblock - 1) / rowsperblock);

        err = cudaMemsetAsync(y_partial_local[i], 0, num_rows_per_device[i] * threadsPerRow * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpyAsync(x_local[i], x, num_cols * sizeof(double), cudaMemcpyDeviceToDevice, s[i]); cuda_err_check(err, __FILE__, __LINE__);

        // Launch the kernel
        gemv_tiled_kernel<<<gridDim, blockDim, sharedMemSize, s[i]>>>(local_A[i], x_local[i], y_partial_local[i], num_rows_per_device[i], num_cols);
        reduce_rows<<<(num_rows_per_device[i] + threadsPerRow - 1) / threadsPerRow, threadsPerRow, 0, s[i]>>>(y_partial_local[i], y_local[i], num_rows_per_device[i], threadsPerRow);
    
        err = cudaMemcpyAsync(y + i * (num_rows_per_device[i]), y_local[i], num_rows_per_device[i] * sizeof(double), cudaMemcpyDeviceToDevice, s[i]); cuda_err_check(err, __FILE__, __LINE__);
    }

    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamSynchronize(s[i]); cuda_err_check(err, __FILE__, __LINE__);
    }

    err = cudaSetDevice(0); cuda_err_check(err, __FILE__, __LINE__);
}



void par_conjugate_gradients_multi_gpu(const double * h_A, const double * h_b, double * h_x, size_t size, int max_iters, double rel_error)
{
    cudaError_t err;

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
    d_local_A_transposed = (const double**)malloc(number_of_devices * sizeof(double*));
    number_of_rows_per_device = (size_t*)malloc(number_of_devices * sizeof(size_t));

    double ** y_partial_local = (double**)malloc(number_of_devices * sizeof(double*));
    double ** y_local = (double**)malloc(number_of_devices * sizeof(double*));
    double ** x_local = (double**)malloc(number_of_devices * sizeof(double*));

    omp_set_num_threads(number_of_devices);


    size_t sharedMemSize = SHMEM;
    size_t threadsPerRow = ((size * sizeof(double)) + sharedMemSize - 1) / sharedMemSize;


    #pragma omp parallel for
    for(size_t i = 0; i < (size_t)number_of_devices; i++)
    {   
        number_of_rows_per_device[i] = (i == (size_t)number_of_devices - 1) ? size - i * (size / (size_t)number_of_devices) : size / (size_t)number_of_devices;
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&d_local_A[i], size * number_of_rows_per_device[i] * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&d_local_A_transposed[i], size * number_of_rows_per_device[i] * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpyAsync((void*)(d_local_A[i]), h_A + i * (size / (size_t)number_of_devices) * size, size * number_of_rows_per_device[i] * sizeof(double), cudaMemcpyHostToDevice, s[i]); cuda_err_check(err, __FILE__, __LINE__);
        // err = cudaMemcpyAsync((void*)d_local_A_transposed[i], (void*)d_local_A[i], size * number_of_rows_per_device[i] * sizeof(double), cudaMemcpyDeviceToDevice, s[i]); cuda_err_check(err, __FILE__, __LINE__);
        transpose<<<dim3(size / TILE_DIM + 1, size / TILE_DIM + 1), dim3(TILE_DIM, TILE_DIM), 0, s[i]>>>((double*)d_local_A_transposed[i], d_local_A[i], number_of_rows_per_device[i], size);

        err = cudaMallocAsync((void**)&y_partial_local[i], number_of_rows_per_device[i] * threadsPerRow * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&y_local[i], number_of_rows_per_device[i] * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&x_local[i], size * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
    }
    
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

    // sync all streams
    for(int i = 0; i < number_of_devices; i++) { err = cudaStreamSynchronize(s[i]); cuda_err_check(err, __FILE__, __LINE__); err = cudaFreeAsync((void*)d_local_A[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);}

    unsigned long long int start, end;

    err = cudaSetDevice(0); cuda_err_check(err, __FILE__, __LINE__);

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    
    bb = dot_kernel_launcher(d_b, d_b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
        // gemv(1.0, A, p, 0.0, Ap, size, size);
        gemv_multi_gpu_tiled_kernel_launcher(d_local_A_transposed, d_p, d_Ap, y_partial_local, y_local, x_local, sharedMemSize, threadsPerRow, number_of_rows_per_device, size, s);
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

    end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    printf("MGPU CG net time: %llums\n", end - start);

    transfer_to_host(d_x, h_x, size);

    // err = cudaFree((void*)d_A); cuda_err_check(err, __FILE__, __LINE__);
    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFreeAsync((void*)d_local_A_transposed[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFreeAsync(y_partial_local[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFreeAsync(y_local[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFreeAsync(x_local[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamDestroy(s[i]); cuda_err_check(err, __FILE__, __LINE__);
    }

    err = cudaFree((void*)d_b); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_r); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_p); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_Ap); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_x); cuda_err_check(err, __FILE__, __LINE__);

    free(s);
    free(d_local_A);
    free(d_local_A_transposed);
    free(number_of_rows_per_device);


    free(y_partial_local);
    free(y_local);
    free(x_local);

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