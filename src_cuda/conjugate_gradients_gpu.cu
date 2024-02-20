#ifndef GPU_LUCA_HPP
#define GPU_LUCA_HPP

#include <cstdio>
#include <cuda_runtime.h>

namespace luca {
void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
        exit (EXIT_FAILURE);
    }
}

__global__ void dot_kernel(const double * x, const double * y, double * result, size_t size)
{
    __shared__ double cache[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0.0;
    while(tid < size)
    {
        temp += __ldg(&x[tid]) * __ldg(&y[tid]);
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while(i != 0)
    {
        if(cacheIndex < i)
        {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0)
    {
        atomicAdd(result, cache[0]);
    }
}

double dot_kernel_launcher(const double * d_x, const double * d_y, size_t size)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    double * d_result, result;

    cudaError_t err;

    err = cudaMalloc((void**)&d_result, sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(d_result, 0, sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    dot_kernel<<<grid_size, block_size>>>(d_x, d_y, d_result, size);

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_result); cuda_err_check(err, __FILE__, __LINE__);

    return result;
}

__global__ void axpby_kernel(double alpha, const double * x, double beta, double * y, size_t size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < size)
        y[tid] = fma(alpha, x[tid], beta * y[tid]);
}

void axpby_kernel_launcher(double alpha, const double * x, double beta, double * y, size_t size)
{
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    axpby_kernel<<<grid_size, block_size>>>(alpha, x, beta, y, size);
    
    cudaError_t err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
}

__global__ void gemv_kernel(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < num_rows)
    {
        double y_val = 0.0;
        for(size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[row + c * num_rows] * x[c];
        }
        y[row] = fma(beta, y[row], y_val);
    }
}

void gemv_kernel_launcher(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;

    gemv_kernel<<<grid_size, block_size>>>(alpha, A, x, beta, y, num_rows, num_cols);

    cudaError_t err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
}

__global__ void gemv_tiled_kernel (const double * a, const double * x, double * y, int m, int n){
    extern __shared__ double work[];
    int global_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_id_y = blockIdx.y * blockDim.y + threadIdx.y;
    int ncols = n / gridDim.x;
    int col0 = ncols * global_id_x; // first value to load
    for (int k = 0; k < ncols; k += blockDim.y)
    {
        int col = k + threadIdx.y;
        if (col < ncols && col0 + col < n) work[col] = x[col0 + col];
    }
    __syncthreads(); // sync group

    if (global_id_y >= m) return;

    double sum = 0;
    for (int k = 0; k < ncols; k++)
    {
        sum += a[global_id_y + m * (col0 + k)] * work[k];
    }
    // if last block and ncols is not multiple of blockDim.y
    if (blockIdx.x == gridDim.x - 1 && n % gridDim.x != 0)
    {
        for (int k = ncols; col0 + k < n; k++)
        {
            sum += a[global_id_y + m * (col0 + k)] * x[col0 + k];
        }
    }
    y[global_id_y + m * global_id_x] = sum;
}

__global__ void reduce_rows(double * y_partial, double * y, int m, int p)
{
    int global_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id_x >= m) return;
    double sum = 0;
    for (int k = 0; k < p; k++)
    {
        sum += y_partial[global_id_x + m * k];
    }
    y[global_id_x] = sum;
}

void gemv_tiled_kernel_launcher(const double * A, const double * x, double * y, size_t num_rows, size_t num_cols)
{
    cudaError_t err;
    int threadsPerRow = 10;
    int rowsperblock = 1024;
    // Define the size of the grid and blocks
    dim3 blockDim(1, rowsperblock);
    dim3 gridDim(threadsPerRow, (num_rows + rowsperblock - 1) / rowsperblock);

    // Calculate the size of the shared memory
    size_t sharedMemSize = num_cols / threadsPerRow * sizeof(double);

    double * y_partial;

    err = cudaMalloc((void**)&y_partial, num_rows * threadsPerRow * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(y_partial, 0, num_rows * threadsPerRow * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    // Launch the kernel
    gemv_tiled_kernel<<<gridDim, blockDim, sharedMemSize>>>(A, x, y_partial, num_rows, num_cols);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);

    // Reduce the rows
    reduce_rows<<<(num_rows + threadsPerRow - 1) / threadsPerRow, threadsPerRow>>>(y_partial, y, num_rows, threadsPerRow);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(y_partial); cuda_err_check(err, __FILE__, __LINE__);
}


void transfer_to_host(const double * d_x, double * h_x, size_t size)
{
    cudaError_t err;

    err = cudaMemcpy(h_x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
}

void par_conjugate_gradients(const double * h_A, const double * h_b, double * h_x, size_t size, int max_iters, double rel_error)
{
    const double * d_A, * d_b;
    int num_iters;

    double alpha, beta, bb, rr, rr_new;
    double * d_r, * d_p, * d_Ap, * d_x;

    cudaError_t err;

    err = cudaMalloc((void**)&d_A, size * size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_b, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMalloc((void**)&d_r, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_p, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_Ap, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMalloc((void**)&d_x, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy((void*)d_A, h_A, size * size * sizeof(double), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy((void*)d_b, h_b, size * sizeof(double), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemset(d_x, 0, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_r, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemcpy(d_p, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);

    bb = dot_kernel_launcher(d_b, d_b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // gemv(1.0, A, p, 0.0, Ap, size, size);
        gemv_tiled_kernel_launcher(d_A, d_p, d_Ap, size, size);
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

    err = cudaFree((void*)d_A); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree((void*)d_b); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_r); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_p); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_Ap); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_x); cuda_err_check(err, __FILE__, __LINE__);

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