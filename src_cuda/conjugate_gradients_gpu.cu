#ifndef GPU_LUCA_HPP
#define GPU_LUCA_HPP

#include <cstdio>
#include <utils.cuh>
#include <chrono>
#include <functional>

#define SHMEM 800

namespace luca {

__global__ void dot_kernel(const double * x, const double * y, double * result, size_t size)
{
    __shared__ double cache[256];
    size_t tid = (size_t)threadIdx.x + (size_t)blockIdx.x * (size_t)blockDim.x;
    size_t cacheIndex = (size_t)threadIdx.x;

    double temp = 0.0;
    while(tid < size)
    {
        temp += __ldg(&x[tid]) * __ldg(&y[tid]);
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    __syncthreads();

    size_t i = blockDim.x / 2;
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
    size_t block_size = 256;
    size_t grid_size = (size + block_size - 1) / block_size;

    double * d_result, result;

    cudaError_t err;

    err = cudaMalloc((void**)&d_result, sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaMemset(d_result, 0, sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    dot_kernel<<<grid_size, block_size>>>(d_x, d_y, d_result, size);

    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);

    err = cudaFree(d_result); cuda_err_check(err, __FILE__, __LINE__);

    return result;
}

__global__ void axpby_kernel(double alpha, const double * x, double beta, double * y, size_t size)
{
    size_t tid = (size_t)threadIdx.x + (size_t)blockIdx.x * (size_t)blockDim.x;
    if(tid < size)
        y[tid] = fma(alpha, x[tid], beta * y[tid]);
}

void axpby_kernel_launcher(double alpha, const double * x, double beta, double * y, size_t size)
{
    size_t block_size = 256;
    size_t grid_size = (size + block_size - 1) / block_size;

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
    size_t block_size = 256;
    size_t grid_size = (num_rows + block_size - 1) / block_size;

    gemv_kernel<<<grid_size, block_size>>>(alpha, A, x, beta, y, num_rows, num_cols);

    cudaError_t err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);
}

__global__ void gemv_tiled_kernel (const double * a, const double * x, double * y, size_t m, size_t n){
    extern __shared__ double work[];
    size_t global_id_x = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t global_id_y = (size_t)blockIdx.y * (size_t)blockDim.y + (size_t)threadIdx.y;
    size_t ncols = n / (size_t)gridDim.x;
    size_t col0 = ncols * global_id_x; // first value to load

    for (size_t k = 0; k < ncols; k += blockDim.y)
    {
        size_t col = k + threadIdx.y;
        if (col < ncols && col0 + col < n) work[col] = x[col0 + col];
    }
    __syncthreads(); // sync group

    if (global_id_y >= m) return;

    double sum = 0;
    for (size_t k = 0; k < ncols; k++)
    {
        sum += a[global_id_y + m * (col0 + k)] * work[k];
    }
    // if last block and ncols is not multiple of blockDim.y
    if (blockIdx.x == gridDim.x - 1 && n % gridDim.x != 0)
    {
        for (size_t k = ncols; col0 + k < n; k++)
        {
            sum += a[global_id_y + m * (col0 + k)] * x[col0 + k];
        }
    }
    y[global_id_y + m * global_id_x] = sum;
}

__global__ void reduce_rows(double * y_partial, double * y, size_t m, size_t p)
{
    size_t global_id_x = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (global_id_x >= m) return;
    double sum = 0;
    for (size_t k = 0; k < p; k++)
    {
        sum += y_partial[global_id_x + m * k];
    }
    y[global_id_x] = sum;
}

void gemv_tiled_kernel_launcher(const double * A, const double * x, double * y, double * y_partial, size_t sharedMemSize, size_t threadsPerRow, size_t num_rows, size_t num_cols)
{
    cudaError_t err;
    // int threadsPerRow = 10;
    size_t rowsperblock = 1024;
    // Define the size of the grid and blocks
    dim3 blockDim(1, rowsperblock);
    dim3 gridDim(threadsPerRow, (num_rows + rowsperblock - 1) / rowsperblock);

    // Calculate the size of the shared memory
    // size_t sharedMemSize = num_cols / threadsPerRow * sizeof(double);

    err = cudaMemset(y_partial, 0, num_rows * threadsPerRow * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    // Launch the kernel
    gemv_tiled_kernel<<<gridDim, blockDim, sharedMemSize>>>(A, x, y_partial, num_rows, num_cols);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);

    // Reduce the rows
    reduce_rows<<<(num_rows + threadsPerRow - 1) / threadsPerRow, threadsPerRow>>>(y_partial, y, num_rows, threadsPerRow);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaGetLastError(); cuda_err_check(err, __FILE__, __LINE__);

}

size_t autotune_gemv_tiled(const double *d_A, double * d_y, size_t m, size_t n, void (*kernel_launcher)(const double *, const double *, double *, double *, size_t, size_t, size_t, size_t))
{
    size_t best_sharedMemSize = 0;
    double best_executionTime = std::numeric_limits<double>::max();
    double executionTime;

    size_t start, end;

    cudaError_t err;

    double * y_partial;
    double * d_x;

    err = cudaMalloc((void**)&d_x, m * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    double * x_local_host = (double*)malloc(m * sizeof(double));

    // fill with random values
    for (size_t i = 0; i < m; i++)
    {
        x_local_host[i] = (((double)rand() / RAND_MAX) - 0.5) * 20;
    }

    err = cudaMemcpy(d_x, x_local_host, m * sizeof(double), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

    for (size_t sharedMemSize = 400; sharedMemSize < 12000; sharedMemSize += 400)
    {
        size_t threadsPerRow = ((m * sizeof(double)) + sharedMemSize - 1) / sharedMemSize;

        err = cudaMalloc((void**)&y_partial, m * threadsPerRow * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

        start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        // Launch the kernel
        kernel_launcher(d_A, d_x, d_y, y_partial, sharedMemSize, threadsPerRow, m, n);

        end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        executionTime = end - start;
    
        // Check if this configuration is the best so far
        if (executionTime < best_executionTime)
        {
            best_sharedMemSize = sharedMemSize;
            best_executionTime = executionTime;
        }

        err = cudaFree(y_partial); cuda_err_check(err, __FILE__, __LINE__);
    }

    err = cudaFree(d_x); cuda_err_check(err, __FILE__, __LINE__);

    printf("Best shared memory size: %lu\n", best_sharedMemSize);

    return best_sharedMemSize;
}



void transfer_to_host(const double * d_x, double * h_x, size_t size)
{
    cudaError_t err;

    err = cudaMemcpy(h_x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaDeviceSynchronize(); cuda_err_check(err, __FILE__, __LINE__);
}

void par_conjugate_gradients(const double * h_A, const double * h_b, double * h_x, size_t size, int max_iters, double rel_error)
{
    fprintf(stderr, "Running parallel CG\n");

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

    double * y_partial = NULL;

    // size_t sharedMemSize = SHMEM;
    size_t sharedMemSize = autotune_gemv_tiled(d_A, d_Ap, size, size, gemv_tiled_kernel_launcher);
    size_t threadsPerRow = ((size * sizeof(double)) + sharedMemSize - 1) / sharedMemSize;

    err = cudaMalloc((void**)&y_partial, size * threadsPerRow * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

    unsigned long long int start, end;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    bb = dot_kernel_launcher(d_b, d_b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // gemv(1.0, A, p, 0.0, Ap, size, size);
        gemv_tiled_kernel_launcher(d_A, d_p, d_Ap, y_partial, sharedMemSize, threadsPerRow, size, size);
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

    printf("Luca CG net time: %llums\n", end - start);

    transfer_to_host(d_x, h_x, size);

    err = cudaFree((void*)d_A); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree((void*)d_b); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_r); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_p); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_Ap); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(d_x); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFree(y_partial); cuda_err_check(err, __FILE__, __LINE__);

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