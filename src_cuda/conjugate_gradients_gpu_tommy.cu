#ifndef GPU_TOMMY_HPP
#define GPU_TOMMY_HPP

#include <iostream>
#include <chrono>
#include <conjugate_gradients_gpu.cu>

#define GRID_SIZE 350
#define BLOCK_SIZE 512


namespace tommy {

void check_cuda(const std::string& msg) {
    cudaDeviceSynchronize();
    cudaError_t err;
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cout << "cuda error: " << msg << std::endl;
        std::cout << "description: " << err << std::endl;
    }
}


template<int blockSize>
__device__ void warpReduce(volatile double* sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template<int blockSize>
__device__ void reduce(double* sdata, int tid) {
    if (blockSize >= 1024) {
        if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
}


template<int blockSize>
__device__ void row_column_mult(const double* __restrict__ A, unsigned int row, int size, const double* __restrict__ p, double* __restrict__ Ap) {
    __shared__ double sArr[blockSize];
    __shared__ double partial;
    if(threadIdx.x == 0) {
        partial = 0.0;
    }
    for(unsigned int i = threadIdx.x; i < size + threadIdx.x; i+=2*blockSize) {
        sArr[threadIdx.x] = ((i<size)?A[row*size + i]*p[i]:0.0) + ((i + blockSize<size)?A[row*size + i + blockSize]*(p[i + blockSize]):0.0);
        __syncthreads();
        reduce<blockSize>(sArr, threadIdx.x);
        if(threadIdx.x == 0) {
            partial += sArr[0];
        }
    }
    if(threadIdx.x == 0) {
        Ap[row] = partial;
    }

}

template<int gridSize, int blockSize>
__global__ void matrix_vector_kernel(const double* __restrict__ A, double* __restrict__ p, double* __restrict__ Ap, int size) {
    for(unsigned int i = blockIdx.x; i < size; i+=gridSize) {
        row_column_mult<blockSize>(A,i,size,p,Ap);
    }

}

template<int gridSize, int blockSize>
void matrix_vector_mult(const double* __restrict__ A, double* __restrict__ p, double* __restrict__ Ap, int size, cudaStream_t stream) {
    matrix_vector_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(A, p, Ap, size);
}


template<int blockSize>
__global__ void sumArray(const double* __restrict__ array, int size, double* __restrict__ result) {
    __shared__ double sArr[blockSize];
    __shared__ double partial;
    if(threadIdx.x == 0) {
        partial = 0;
    }
    for(unsigned int i = threadIdx.x; i < size + threadIdx.x; i+=2*blockSize) {
        sArr[threadIdx.x] = ((i<size)?array[i]:0.0) + ((i + blockSize < size)?array[i + blockSize]:0.0);
        __syncthreads();
        reduce<blockSize>(sArr, threadIdx.x);
        if(threadIdx.x == 0) {
            partial += sArr[0];
        }

    }
    if(threadIdx.x == 0) {
        *result = partial;
    }
}

template<int gridSize, int blockSize>
__global__ void dot_product_kernel(const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ outArray, int size) {
    __shared__ double sArr[blockSize];
    if(threadIdx.x == 0) {
        outArray[blockIdx.x] = 0.0;
    }
    for(unsigned int i = blockIdx.x; 2*blockSize*i < size; i+=gridSize) {
        int tmp = i*2*blockSize + threadIdx.x;
        sArr[threadIdx.x] = ((tmp<size)?x[tmp]*y[tmp]:0.0) + ((tmp + blockSize<size)?x[tmp + blockSize]*y[tmp + blockSize]:0.0);
        __syncthreads();
        reduce<blockSize>(sArr, threadIdx.x);
        if(threadIdx.x == 0) {
            outArray[blockIdx.x] += sArr[0];
        }
    }
}

template<int gridSize, int blockSize>
void dot_product(const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ outArray, int size, double* __restrict__ result, cudaStream_t stream) {
    dot_product_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(x, y, outArray, size);
    sumArray<blockSize><<<1, blockSize>>>(outArray, gridSize, result);
}

template<int gridSize, int blockSize>
__global__ void axpby_kernel(const double* __restrict__ alpha, const double* __restrict__ x, double* __restrict__ y, int size) {
    int th_id = threadIdx.x + blockIdx.x * blockSize;
    if(th_id < size) {
        y[th_id] = (*alpha) * x[th_id] + y[th_id];
    }
}


template<int gridSize, int blockSize>
__global__ void _minus_axpby_kernel(const double* __restrict__ alpha, const double* __restrict__ x, double* __restrict__ y, int size) {
    int th_id = threadIdx.x + blockIdx.x * blockSize;
    if(th_id < size) {
        y[th_id] = -(*alpha) * x[th_id] + y[th_id];
    }
}

template<int gridSize, int blockSize>
__global__ void xpby_kernel( const double* __restrict__ x, double* __restrict__ y, const double* __restrict__ beta, int size) {
    int th_id = threadIdx.x + blockIdx.x * blockSize;
    if(th_id < size) {
        y[th_id] = (x[th_id] + (*beta) * y[th_id]);
    }
}



__global__ void divide(const double* __restrict__ div1, const double* __restrict__ div2, double* result) {
    if(threadIdx.x == 0) {
        *result = *div1 / *div2;
    }

}

void matrix_vector(double* matrix, double* vector, double* sol, int size) {
    for(int i = 0; i < size; i++) {
        sol[i] = 0;
        for(int j = 0; j < size; j++) {
            sol[i] += matrix[i*size + j] * vector[j];
        }
    }
}

template<int gridSize, int blockSize>
void axpby(double* __restrict__ alpha, const double * __restrict__ x, double * __restrict__ y, int size, cudaStream_t stream)
{
    axpby_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(alpha, x, y, size);
}

template<int gridSize, int blockSize>
void _minus_axpby(double* __restrict__ alpha, const double * __restrict__ x, double * __restrict__ y, int size, cudaStream_t stream)
{
    _minus_axpby_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(alpha, x, y, size);
}


template<int gridSize, int blockSize>
void xpby(const double * __restrict__ x, double * __restrict__ y, const double* __restrict__ beta, int size, cudaStream_t stream)
{
    xpby_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(x, y, beta, size);
}


// template<char use_original = 1>
void conjugate_gradients(const double * h_A, const double * h_b, double * h_x, size_t size, int max_iters, double rel_error) {
    double* r_cuda;
    double* p_cuda;
    double* Ap_cuda;
    double *alpha;
    double *beta;
    double* bb;
    double bb_cpu;
    double* rr;
    double* rr_new;
    double* dot_product_out_array;
    double err;

    double * A, * x, * b;

    cudaMalloc(&r_cuda, size*sizeof(double));
    cudaMalloc(&p_cuda, size*sizeof(double));
    cudaMalloc(&Ap_cuda, size*sizeof(double));
    cudaMalloc(&dot_product_out_array, sizeof(double)*GRID_SIZE);
    cudaMalloc(&alpha, sizeof(double));
    cudaMalloc(&beta, sizeof(double));
    cudaMalloc(&bb, sizeof(double));
    cudaMalloc(&rr, sizeof(double));
    cudaMalloc(&rr_new, sizeof(double));
    cudaMalloc(&A, size*size*sizeof(double));
    cudaMalloc(&x, size*sizeof(double));
    cudaMalloc(&b, size*sizeof(double));

    cudaMemcpy(r_cuda, h_b, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(p_cuda, h_b, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A, h_A, size*size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x, h_x, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(b, h_b, size*sizeof(double), cudaMemcpyHostToDevice);


    int niters;
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    dot_product<GRID_SIZE, BLOCK_SIZE>(b, b, dot_product_out_array, (int) size, bb, stream1);
    cudaMemcpy(&bb_cpu, bb, sizeof(double), cudaMemcpyDeviceToHost);
    err = bb_cpu;
    cudaMemcpy(rr, bb, sizeof(double), cudaMemcpyDeviceToDevice);
    for(niters = 1; niters <= max_iters; niters++) {
        // if constexpr (use_original) {
            matrix_vector_mult<GRID_SIZE, BLOCK_SIZE>(A, p_cuda, Ap_cuda, (int)size, stream1);
        // } else {
        //     luca::gemv_tiled_kernel_launcher(A, p_cuda, Ap_cuda, size, size);
        // }
        dot_product<GRID_SIZE, BLOCK_SIZE>(p_cuda, Ap_cuda, dot_product_out_array,(int)size, alpha, stream1);
        divide<<<1,1, 0, stream1>>>(rr,alpha, alpha);
        axpby<GRID_SIZE, BLOCK_SIZE>(alpha, p_cuda, x, (int)size, stream1);
        _minus_axpby<GRID_SIZE, BLOCK_SIZE>(alpha, Ap_cuda, r_cuda, (int) size, stream1);
        dot_product<GRID_SIZE, BLOCK_SIZE>(r_cuda, r_cuda, dot_product_out_array, (int)size, rr_new, stream1);
        divide<<<1, 1, 0, stream1>>>(rr_new, rr, beta);
        cudaMemcpy(rr, rr_new, sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&err, rr, sizeof(double), cudaMemcpyDeviceToHost);
        if(std::sqrt(err / bb_cpu) < rel_error) { break; }
        xpby<GRID_SIZE, BLOCK_SIZE>(r_cuda, p_cuda, beta,  (int)size, stream1);
    }
    if(niters < max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", niters, std::sqrt(err / bb_cpu));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(err / bb_cpu));
    }

    cudaMemcpy(h_x, x, size*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(r_cuda);
    cudaFree(p_cuda);
    cudaFree(Ap_cuda);
    cudaFree(dot_product_out_array);
    cudaFree(alpha);
    cudaFree(beta);
    cudaFree(bb);
    cudaFree(rr);
    cudaFree(rr_new);
}
}
#endif //GPU_TOMMY_HPP