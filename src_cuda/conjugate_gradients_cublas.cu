#ifndef CUBLAS_CG_H
#define CUBLAS_CG_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <utils.cuh>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

void transfer_to_host(const double * d_x, double * h_x, size_t size)
{
    CUDA_CHECK(cudaMemcpy(h_x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void conjugate_gradients_cublas(const double * h_A, const double * h_b, double * h_x, size_t size, int max_iters, double rel_error)
{
    const double * d_A, * d_b;
    int num_iters;

    double alpha, beta, bb, rr, rr_new;
    double * d_r, * d_p, * d_Ap, * d_x;

    CUDA_CHECK(cudaMalloc((void**)&d_A, size * size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size * sizeof(double)));

    CUDA_CHECK(cudaMalloc((void**)&d_r, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_p, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Ap, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, size * sizeof(double)));

    CUDA_CHECK(cudaMemcpy((void*)d_A, h_A, size * size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)d_b, h_b, size * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_x, 0, size * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_r, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_p, d_b, size * sizeof(double), cudaMemcpyDeviceToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const double one = 1.0;
    const double zero = 0.0;

    cublasDdot(handle, size, d_b, 1, d_b, 1, &bb);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        // gemv(1.0, A, p, 0.0, Ap, size, size);
        CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, size, size, &one, d_A, size, d_p, 1, &zero, d_Ap, 1));
        // alpha = rr / dot(p, Ap, size);
        CUBLAS_CHECK(cublasDdot(handle, size, d_p, 1, d_Ap, 1, &alpha));
        alpha = rr / alpha;
        // axpby(alpha, p, 1.0, x, size);
        CUBLAS_CHECK(cublasDaxpy(handle, size, &alpha, d_p, 1, d_x, 1));
        // axpby(-alpha, Ap, 1.0, r, size);
        alpha = -alpha;
        CUBLAS_CHECK(cublasDaxpy(handle, size, &alpha, d_Ap, 1, d_r, 1));
        // rr_new = dot(r, r, size);
        CUBLAS_CHECK(cublasDdot(handle, size, d_r, 1, d_r, 1, &rr_new));
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
        // axpby(1.0, r, beta, p, size);
        CUBLAS_CHECK(cublasDscal(handle, size, &beta, d_p, 1));
        double one = 1.0;
        CUBLAS_CHECK(cublasDaxpy(handle, size, &one, d_r, 1, d_p, 1));
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    transfer_to_host(d_x, h_x, size);

    CUDA_CHECK(cudaFree((void*)d_A));
    CUDA_CHECK(cudaFree((void*)d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_x));

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
    }
}

#endif