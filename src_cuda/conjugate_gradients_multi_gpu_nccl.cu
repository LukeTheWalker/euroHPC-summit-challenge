#ifndef MULTI_GPU_NCCL_LUCA_HPP
#define MULTI_GPU_NCCL_LUCA_HPP

#include <cstdio>
#include <utils.cuh>
#include <conjugate_gradients_gpu.cu>
#include <conjugate_gradients_multi_gpu.cu>
#include <nccl.h>
#include <omp.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

extern int myRank, nRanks;
extern ncclComm_t * comms;

#define SHMEM 800

namespace luca {

void gemv_multi_gpu_nccl_tiled_kernel_launcher(const double ** local_A, const double * x, double * y, double ** y_partial_local, double ** y_local, double ** x_local, size_t * num_rows_per_device, size_t * num_rows_per_node, size_t num_cols, cudaStream_t * s)
{
    size_t number_of_devices; cudaError_t err; ncclResult_t nccl_err;

    err = cudaGetDeviceCount((int*)&number_of_devices); cuda_err_check(err, __FILE__, __LINE__);

    int threadsPerRow = 10;
    size_t sharedMemSize = num_cols / threadsPerRow * sizeof(double);

    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);

        int rowsperblock = 1024;
        // Define the size of the grid and blocks
        dim3 blockDim(1, rowsperblock);
        dim3 gridDim(threadsPerRow, (num_rows_per_device[i] + rowsperblock - 1) / rowsperblock);

        err = cudaMemsetAsync(y_partial_local[i], 0, num_rows_per_device[i] * threadsPerRow * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);

        nccl_err = ncclGroupStart(); nccl_err_check(nccl_err, __FILE__, __LINE__);
        ncclBroadcast(x, x_local[i], num_cols, ncclDouble, 0, comms[i], s[i]);
        nccl_err = ncclGroupEnd(); nccl_err_check(nccl_err, __FILE__, __LINE__);

        // Launch the kernel
        gemv_tiled_kernel<<<gridDim, blockDim, sharedMemSize, s[i]>>>(local_A[i], x_local[i], y_partial_local[i], num_rows_per_device[i], num_cols);
        reduce_rows<<<(num_rows_per_device[i] + threadsPerRow - 1) / threadsPerRow, threadsPerRow, 0, s[i]>>>(y_partial_local[i], y_local[i], num_rows_per_device[i], threadsPerRow);
    }

    nccl_err = ncclGroupStart(); nccl_err_check(nccl_err, __FILE__, __LINE__);

    for (int i = 0; i < number_of_devices; i++)
        nccl_err = ncclSend(y_local[i], num_rows_per_device[i], ncclDouble, 0, comms[i], s[i]); nccl_err_check(nccl_err, __FILE__, __LINE__);
    
    if (myRank == 0){
        size_t progressive_offset = 0;
        for (size_t r = 0; r < nRanks; r++){
            for (size_t i = 0; i < number_of_devices; i++){
                size_t num_to_transfer = (i == number_of_devices - 1) ? num_rows_per_node[r] - i * (num_rows_per_node[r] / number_of_devices) : num_rows_per_node[r] / number_of_devices;
                nccl_err = ncclRecv(y + progressive_offset, num_to_transfer, ncclDouble, r * number_of_devices + i, comms[0], s[0]); nccl_err_check(nccl_err, __FILE__, __LINE__);
                progressive_offset += num_to_transfer;
            }
        }
    }
    
    nccl_err = ncclGroupEnd(); nccl_err_check(nccl_err, __FILE__, __LINE__);

    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamSynchronize(s[i]); cuda_err_check(err, __FILE__, __LINE__);
    }

    if (myRank == 0)
    err = cudaSetDevice(0); cuda_err_check(err, __FILE__, __LINE__);
}



void par_conjugate_gradients_multi_gpu_nccl(const double * h_A, const double * h_b, double * h_x, size_t size, int max_iters, double rel_error)
{
    cudaError_t err;

    int number_of_devices;
    cudaStream_t * s;
    size_t * number_of_rows_per_device, *number_of_rows_per_node;
    const double ** d_local_A, ** d_local_A_transposed;

    err = cudaGetDeviceCount(&number_of_devices); cuda_err_check(err, __FILE__, __LINE__);
    s = (cudaStream_t*)malloc(number_of_devices * sizeof(cudaStream_t));
    d_local_A = (const double**)malloc(number_of_devices * sizeof(const double*));
    d_local_A_transposed = (const double**)malloc(number_of_devices * sizeof(double*));
    number_of_rows_per_device = (size_t*)malloc(number_of_devices * sizeof(size_t));
    number_of_rows_per_node = (size_t*)malloc(sizeof(size_t) * nRanks);

    double ** y_partial_local = (double**)malloc(number_of_devices * sizeof(double*));
    double ** y_local = (double**)malloc(number_of_devices * sizeof(double*));
    double ** x_local = (double**)malloc(number_of_devices * sizeof(double*));

    for (int i = 0; i < nRanks; i++) number_of_rows_per_node[i] = (i == nRanks - 1) ? size - i * (size / nRanks) : size / nRanks;
    if (myRank == 0) for (int i = 0; i < nRanks; i++) fprintf(stderr, "Rank %d, number of rows: %ld\n", i, number_of_rows_per_node[i]);

    omp_set_num_threads(number_of_devices);

    #pragma omp parallel for
    for(int i = 0; i < number_of_devices; i++)
    {
        number_of_rows_per_device[i] = (i == number_of_devices - 1) ? number_of_rows_per_node[myRank] - i * (number_of_rows_per_node[myRank] / number_of_devices) : number_of_rows_per_node[myRank] / number_of_devices;
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&d_local_A[i], size * number_of_rows_per_device[i] * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&d_local_A_transposed[i], size * number_of_rows_per_device[i] * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        
        size_t offset =  myRank * (size / nRanks) * size + i * (size / nRanks / number_of_devices) * size;
        err = cudaMemcpyAsync((void*)(d_local_A[i]), h_A + offset, size * number_of_rows_per_device[i] * sizeof(double), cudaMemcpyHostToDevice, s[i]); cuda_err_check(err, __FILE__, __LINE__);

        transpose<<<dim3(size / TILE_DIM + 1, size / TILE_DIM + 1), dim3(TILE_DIM, TILE_DIM), 0, s[i]>>>((double*)d_local_A_transposed[i], d_local_A[i], number_of_rows_per_device[i], size);

        size_t rowsperblock = 1024;
        size_t sharedMemSize = SHMEM;
        size_t threadsPerRow = ((size * sizeof(double)) + sharedMemSize - 1) / sharedMemSize;


        err = cudaMallocAsync((void**)&y_partial_local[i], number_of_rows_per_device[i] * threadsPerRow * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&y_local[i], number_of_rows_per_device[i] * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMallocAsync((void**)&x_local[i], size * sizeof(double), s[i]); cuda_err_check(err, __FILE__, __LINE__);

    }

    const double * d_b;
    int num_iters;

    double alpha = 0, beta = 0, rr = 0, rr_new = 0, bb = 0;
    double * d_r, * d_p, * d_Ap, * d_x;

    if (myRank == 0)
    {
        err = cudaSetDevice(0); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMalloc((void**)&d_b, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMalloc((void**)&d_r, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc((void**)&d_p, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc((void**)&d_Ap, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMalloc((void**)&d_x, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy((void*)d_b, h_b, size * sizeof(double), cudaMemcpyHostToDevice); cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemset(d_x, 0, size * sizeof(double)); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_r, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemcpy(d_p, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice); cuda_err_check(err, __FILE__, __LINE__);
    }

    if (myRank == 0)
    {
        bb = dot_kernel_launcher(d_b, d_b, size);
        rr = bb;
    }
    bool done = false; int mpi_err;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        mpi_err = MPI_Bcast(&done, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD); mpi_err_check(mpi_err, __FILE__, __LINE__);
        if (done) { break; }
        // gemv(1.0, A, p, 0.0, Ap, size, size);
        gemv_multi_gpu_nccl_tiled_kernel_launcher(d_local_A_transposed, d_p, d_Ap, y_partial_local, y_local, x_local, number_of_rows_per_device, number_of_rows_per_node, size, s);
        // alpha = rr / dot(p, Ap, size);
        if (myRank == 0) {
            alpha = rr / dot_kernel_launcher(d_p, d_Ap, size);
            // axpby(alpha, p, 1.0, x, size);
            axpby_kernel_launcher(alpha, d_p, 1.0, d_x, size);
            // axpby(-alpha, Ap, 1.0, r, size);
            axpby_kernel_launcher(-alpha, d_Ap, 1.0, d_r, size);
            // rr_new = dot(r, r, size);
            rr_new = dot_kernel_launcher(d_r, d_r, size);
            beta = rr_new / rr;
            rr = rr_new;
            if(std::sqrt(rr / bb) < rel_error) { done = true; }
            // axpby(1.0, r, beta, p, size);
            axpby_kernel_launcher(1.0, d_r, beta, d_p, size);
        }
    }

    if (myRank == 0)
        transfer_to_host(d_x, h_x, size);

    for (int i = 0; i < number_of_devices; i++)
        nccl_err_check(ncclCommDestroy(comms[i]), __FILE__, __LINE__);

    for (int i = 0; i < number_of_devices; i++)
    {
        err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFreeAsync((void*)d_local_A_transposed[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFreeAsync((void*)d_local_A[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFreeAsync(y_partial_local[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFreeAsync(y_local[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFreeAsync(x_local[i], s[i]); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaStreamDestroy(s[i]); cuda_err_check(err, __FILE__, __LINE__);
    }

    if (myRank == 0){    
        err = cudaFree((void*)d_b); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_r); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_p); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_Ap); cuda_err_check(err, __FILE__, __LINE__);
        err = cudaFree(d_x); cuda_err_check(err, __FILE__, __LINE__);
    }
    
    free(s);
    free(d_local_A);
    free(d_local_A_transposed);
    free(number_of_rows_per_device);
    free(number_of_rows_per_node);
    free(comms);

    free(y_partial_local);
    free(y_local);
    free(x_local);


    if (myRank == 0){
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
}

#endif