
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#define GRID_SIZE 1000
#define BLOCK_SIZE 512
#define WARP_SIZE 32




namespace luca {
    void cuda_err_check(cudaError_t err, const char *file, int line) {
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), file, line);
            exit(EXIT_FAILURE);
        }
    }

    __global__ void dot_kernel(const double *x, const double *y, double *result, size_t size) {
        __shared__ double cache[256];
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int cacheIndex = threadIdx.x;

        double temp = 0.0;
        while (tid < size) {
            temp += __ldg(&x[tid]) * __ldg(&y[tid]);
            tid += blockDim.x * gridDim.x;
        }

        cache[cacheIndex] = temp;

        __syncthreads();

        int i = blockDim.x / 2;
        while (i != 0) {
            if (cacheIndex < i) {
                cache[cacheIndex] += cache[cacheIndex + i];
            }
            __syncthreads();
            i /= 2;
        }

        if (cacheIndex == 0) {
            atomicAdd(result, cache[0]);
        }
    }

    double dot_kernel_launcher(const double *d_x, const double *d_y, size_t size) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;

        double *d_result, result;

        cudaError_t err;

        err = cudaMalloc((void **) &d_result, sizeof(double));
        cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemset(d_result, 0, sizeof(double));
        cuda_err_check(err, __FILE__, __LINE__);

        dot_kernel<<<grid_size, block_size>>>(d_x, d_y, d_result, size);

        err = cudaGetLastError();
        cuda_err_check(err, __FILE__, __LINE__);

        err = cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
        cuda_err_check(err, __FILE__, __LINE__);

        err = cudaFree(d_result);
        cuda_err_check(err, __FILE__, __LINE__);

        return result;
    }

    __global__ void axpby_kernel(double alpha, const double *x, double beta, double *y, size_t size) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid < size)
            y[tid] = fma(alpha, x[tid], beta * y[tid]);
    }

    void axpby_kernel_launcher(double alpha, const double *x, double beta, double *y, size_t size) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;

        axpby_kernel<<<grid_size, block_size>>>(alpha, x, beta, y, size);

        cudaError_t err = cudaGetLastError();
        cuda_err_check(err, __FILE__, __LINE__);
    }

    __global__ void gemv_kernel(double alpha, const double *A, const double *x, double beta, double *y, size_t num_rows,
                                size_t num_cols) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < num_rows) {
            double y_val = 0.0;
            for (size_t c = 0; c < num_cols; c++) {
                y_val += alpha * A[row + c * num_rows] * x[c];
            }
            y[row] = fma(beta, y[row], y_val);
        }
    }

    void gemv_kernel_launcher(double alpha, const double *A, const double *x, double beta, double *y, size_t num_rows,
                              size_t num_cols) {
        int block_size = 256;
        int grid_size = (num_rows + block_size - 1) / block_size;

        gemv_kernel<<<grid_size, block_size>>>(alpha, A, x, beta, y, num_rows, num_cols);

        cudaError_t err = cudaGetLastError();
        cuda_err_check(err, __FILE__, __LINE__);
    }



    template<int ncols>
    __global__ void gemv_tiled_kernel(const double * __restrict__ a, const double * __restrict__ x, double * __restrict__ y, int m, int n) {
        extern __shared__ double work[];
        int global_id_x = blockIdx.x * blockDim.x + threadIdx.x;
        int global_id_y = blockIdx.y * blockDim.y + threadIdx.y;
        //int ncols = n / gridDim.x;
        int col0 = ncols * global_id_x; // first value to load
        for (int k = 0; k < ncols; k += blockDim.y) {
            int col = k + threadIdx.y;
            if (col < ncols && col0 + col < n) work[col] = x[col0 + col];
        }
        __syncthreads(); // sync group

        if (global_id_y >= m) return;

        double sum = 0;
#pragma unroll
        for (int k = 0; k < ncols; k++) {
            sum += a[global_id_y + m * (col0 + k)] * work[k];
        }
        // if last block and ncols is not multiple of blockDim.y
        if (blockIdx.x == gridDim.x - 1 && n % gridDim.x != 0) {
            for (int k = ncols; col0 + k < n; k++) {
                sum += a[global_id_y + m * (col0 + k)] * x[col0 + k];
            }
        }
        y[global_id_y + m * global_id_x] = sum;
    }

    __global__ void reduce_rows(double *y_partial, double *y, int m, int p) {
        int global_id_x = blockIdx.x * blockDim.x + threadIdx.x;
        if (global_id_x >= m) return;
        double sum = 0;
        for (int k = 0; k < p; k++) {
            sum += y_partial[global_id_x + m * k];
        }
        y[global_id_x] = sum;
    }

    void gemv_tiled_kernel_launcher(const double *A, const double *x, double *y, size_t num_rows, size_t num_cols) {
        cudaError_t err;
        int threadsPerRow = 10;
        int rowsperblock = 1024;
        // Define the size of the grid and blocks
        dim3 blockDim(1, rowsperblock);
        dim3 gridDim(threadsPerRow, (num_rows + rowsperblock - 1) / rowsperblock);

        // Calculate the size of the shared memory
        size_t sharedMemSize = num_cols / threadsPerRow * sizeof(double);

        double *y_partial;

        err = cudaMalloc((void **) &y_partial, num_rows * threadsPerRow * sizeof(double));
        cuda_err_check(err, __FILE__, __LINE__);
        err = cudaMemset(y_partial, 0, num_rows * threadsPerRow * sizeof(double));
        cuda_err_check(err, __FILE__, __LINE__);

        // Launch the kernel
        gemv_tiled_kernel<1000><<<gridDim, blockDim, sharedMemSize>>>(A, x, y_partial, num_rows, num_cols);
        //err = cudaDeviceSynchronize();
        cuda_err_check(err, __FILE__, __LINE__);
        err = cudaGetLastError();
        cuda_err_check(err, __FILE__, __LINE__);

        // Reduce the rows
        reduce_rows<<<(num_rows + threadsPerRow - 1) / threadsPerRow, threadsPerRow>>>(y_partial, y, num_rows,
                                                                                       threadsPerRow);
        //err = cudaDeviceSynchronize();
        cuda_err_check(err, __FILE__, __LINE__);
        err = cudaGetLastError();
        cuda_err_check(err, __FILE__, __LINE__);

        err = cudaFree(y_partial);
        cuda_err_check(err, __FILE__, __LINE__);
    }


    void transfer_to_host(const double *d_x, double *h_x, size_t size) {
        cudaError_t err;

        err = cudaMemcpy(h_x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost);
        cuda_err_check(err, __FILE__, __LINE__);
        err = cudaDeviceSynchronize();
        cuda_err_check(err, __FILE__, __LINE__);
    }
}





#define cublasCheckErrors(fn) \
    do { \
        cublasStatus_t __err = fn; \
        if (__err != CUBLAS_STATUS_SUCCESS) { \
            printf("Fatal cublas error: %d (at %s:%d)\n", \
                (int)(__err), \
                __FILE__, __LINE__); \
            printf("*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


void check_cuda(const std::string& msg) {
    cudaDeviceSynchronize();
    cudaError_t err;
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cout << "cuda error: " << msg << std::endl;
        std::cout << "description: " << err << std::endl;
        exit(1);
    }
}

double dot(const double * x, const double * y, size_t size)
{
    double result = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}



void axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    // y = alpha * x + beta * y

    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
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

void generate_matrix(size_t n, double** matrix_out) {
    auto* matrix = new double[n * n];
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
    auto* rhs = new double[n];
    for(size_t i = 0; i < n; i++) {
        rhs[i] = value;
    }
    *rhs_out = rhs;
}


template<int blockSize>
__device__ void reduce_ws(double* __restrict__ data, float* __restrict__ out) {
    __shared__ float sdata[WARP_SIZE];
    int tid = threadIdx.x;
    float val;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % WARP_SIZE;
    int warpID = threadIdx.x / WARP_SIZE;
    val = data[tid];
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    if (lane == 0){
        sdata[warpID] = val;
    }
    __syncthreads();

    if (warpID == 0){
        val = (tid < blockSize/WARP_SIZE)?sdata[lane]:0;
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (tid == 0) {
            atomicAdd(out, val);
        }
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
__device__ void row_column_mult_ws(const double* __restrict__ A, unsigned int row, int size, const double* __restrict__ p, double* __restrict__ Ap) {
    __shared__ double sArr[blockSize];
    __shared__ float partial;
    if(threadIdx.x == 0) {
        partial = 0.0;
    }
    for(unsigned int i = threadIdx.x; i < size + threadIdx.x; i+=2*blockSize) {
        //sArr[threadIdx.x] = ((i<size)?A[row*size + i]*p[i]:0.0) + ((i + blockSize<size)?A[row*size + i + blockSize]*(p[i + blockSize]):0.0);

        if(i < size && i + blockSize < size) {
            sArr[threadIdx.x] = fma(A[row*size + i],p[i], A[row*size + i + blockSize] * __ldg(&p[i + blockSize]));
        } else if(i < size){
            sArr[threadIdx.x] = A[row*size + i]*p[i];
        } else {
            sArr[threadIdx.x] = 0.0;
        }

        __syncthreads();
        reduce_ws<blockSize>(sArr, &partial);
    }
    if(threadIdx.x == 0) {
        Ap[row] = partial;
    }

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

template<int blockSize>
__global__ void tiled_matrix_vector_mult(const double* __restrict__ A, const double* __restrict__ p, double* __restrict__ Ap, const unsigned int size) {
    __shared__ double sArr[blockSize];
    double Ap_partial = 0;
    const int tid = threadIdx.x + blockSize * blockIdx.x;
    for(unsigned int k = 0; k < (size - 1 + blockSize)/blockSize; k++) {
        sArr[threadIdx.x] = (k*blockSize + threadIdx.x < size) ? p[k*blockSize + threadIdx.x] : 0.0;
        __syncthreads();
        for(unsigned int e = 0; e < blockSize; e++) {
            Ap_partial += (tid + size * (k*blockSize + e) < size*size)?(A[tid + size * (k*blockSize + e)] * sArr[e]):0.0;
        }
        __syncthreads();
    }
    if(tid < size) {
        Ap[tid] = Ap_partial;
    }
}


template<int gridSize, int blockSize>
__global__ void matrix_vector_kernel(const double* __restrict__ A, double* __restrict__ p, double* __restrict__ Ap, int size) {
    for(unsigned int i = blockIdx.x; i < size; i+=gridSize) {
        //row_column_mult<blockSize>(A,i,size,p,Ap);
        row_column_mult_ws<blockSize>(A,i,size,p,Ap);
    }

}

template<int gridSize, int blockSize>
void matrix_vector_mult(const double* __restrict__ A, double* __restrict__ p, double* __restrict__ Ap, int size, cudaStream_t stream) {
    //tiled_matrix_vector_mult<blockSize><<<(size  + blockSize)/blockSize, blockSize>>>(A, p, Ap, size);
    matrix_vector_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(A, p, Ap, size);
    //matrix_vector_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(A, p, Ap, size);
}


template<int gridSize, int blockSize>
void matrix_vector_mult_cublas(cublasHandle_t  handle, const double* __restrict__ A, double* __restrict__ p, double* __restrict__ Ap, int size) {
    //tiled_matrix_vector_mult<blockSize><<<(size  + blockSize)/blockSize, blockSize>>>(A, p, Ap, size);
    //matrix_vector_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(A, p, Ap, size);
    //matrix_vector_kernel<gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(A, p, Ap, size);
    double alpha = 1.0f;
    double beta = 0;
    cublasCheckErrors(cublasDgemv(handle, CUBLAS_OP_N, size, size, &alpha, A, size, p, 1, &beta, Ap, 1));
    check_cuda("cublas error");
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



void conjugate_gradients_serial(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error, long* execution_time)
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
    auto start = std::chrono::high_resolution_clock::now();
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
    auto stop = std::chrono::high_resolution_clock::now();
    *execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

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

void conjugate_gradients(const double * A_cpu, const double * b_cpu, double * x_cpu, size_t size, int max_iters, double rel_error, long* execution_time) {
    auto start = std::chrono::high_resolution_clock::now();
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
    double* A;
    double* b;
    double* x;
    cublasHandle_t handle;
    cublasCheckErrors(cublasCreate(&handle));
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    cudaMalloc(&A, sizeof(double) * size*size);
    cudaMalloc(&b, sizeof(double) * size);

    cudaMalloc(&x, sizeof(double) * size);
    cudaMemcpy(A,A_cpu, sizeof(double)*size*size, cudaMemcpyHostToDevice);
    cudaMemcpy(b,b_cpu, sizeof(double)*size, cudaMemcpyHostToDevice);
    cudaMemcpy(x,x_cpu, sizeof(double)*size, cudaMemcpyHostToDevice);

    cudaMalloc(&r_cuda, size*sizeof(double));
    cudaMalloc(&p_cuda, size*sizeof(double));
    cudaMalloc(&Ap_cuda, size*sizeof(double));

    cudaMalloc(&dot_product_out_array, sizeof(double)*GRID_SIZE);
    cudaMalloc(&alpha, sizeof(double));
    cudaMalloc(&beta, sizeof(double));
    cudaMalloc(&bb, sizeof(double));
    cudaMalloc(&rr, sizeof(double));
    cudaMalloc(&rr_new, sizeof(double));
    cudaMemcpy(r_cuda, b, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(p_cuda, b, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(x,0,sizeof(double) * size);
    int niters;
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    dot_product<GRID_SIZE, BLOCK_SIZE>(b, b, dot_product_out_array, (int) size, bb, stream1);
    cudaMemcpy(&bb_cpu, bb, sizeof(double), cudaMemcpyDeviceToHost);
    err = bb_cpu;
    cudaMemcpy(rr, bb, sizeof(double), cudaMemcpyDeviceToDevice);

    //std::cout << "starting cuda" << std::endl;
    for(niters = 1; niters < max_iters; niters++) {
        //matrix_vector_mult_cublas<GRID_SIZE, BLOCK_SIZE>(handle, A, p_cuda, Ap_cuda, (int)size);
        //std::cout << "finished cuda" << std::endl;
        luca::gemv_tiled_kernel_launcher(A, p_cuda, Ap_cuda, size, size);
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
    cudaFree(A);
    cudaFree(b);
    cudaFree(x);
    cudaFree(r_cuda);
    cudaFree(p_cuda);
    cudaFree(Ap_cuda);
    cudaFree(dot_product_out_array);
    cudaFree(alpha);
    cudaFree(beta);
    cudaFree(bb);
    cudaFree(rr);
    cudaFree(rr_new);
    auto stop = std::chrono::high_resolution_clock::now();
    *execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

}

void print_sol(double* sol) {
    for(int i = 0; i < 5; i++) {
        std::cout << sol[i] << std::endl;
    }
}

void print_sol_cuda(double* sol) {
    double* tmp = new double[5];
    cudaMemcpy(tmp, sol, 5*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 5; i++) {
        std::cout << tmp[i] << std::endl;
    }
}



int main(int argc, char ** argv) {

    int size = 500;
    int max_iters = 5000;
    double rel_error = 1e-9;
    int serial_trials = 1;
    int parallel_trials = 1;
    if(argc > 1) size = atoi(argv[1]);
    if(argc > 2) max_iters = atoi(argv[2]);
    if(argc > 3) rel_error = atof(argv[3]);
    if(argc > 4) serial_trials = atoi(argv[4]);
    if(argc > 5) parallel_trials = atoi(argv[5]);

    printf("Command line arguments:\n");
    printf("  matrix_size: %d\n", size);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("  serial trials number:         %d\n", serial_trials);
    printf("  parallel trials number:         %d\n", parallel_trials);
    printf("\n");

    long serial_execution_time = 0;
    long parallel_execution_time = 0;

    int* size_cuda;
    int* max_iters_cuda;
    double* tol_cuda;
    double* matrix;
    double* rhs;
    double* r_cuda;
    double* p_cuda;
    double* Ap_cuda;
    generate_matrix(size, &matrix);
    generate_rhs(size, 1.0, &rhs);
    auto* sol = new double[size];

    for(int i = 0; i < size; i++) {
        sol[i] = 1.0;
    }


    cudaMalloc(&max_iters_cuda, sizeof(int));
    cudaMalloc(&size_cuda, sizeof(int));
    cudaMalloc(&tol_cuda, sizeof(double));
    cudaMalloc(&r_cuda, size*sizeof(double));
    cudaMalloc(&p_cuda, size*sizeof(double));
    cudaMalloc(&Ap_cuda, size*sizeof(double));
    cudaMemcpy(max_iters_cuda, &max_iters, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(size_cuda, &size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tol_cuda, &rel_error, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(r_cuda, rhs, size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(p_cuda, rhs, size*sizeof(double), cudaMemcpyHostToDevice);




    for(int i = 0; i < serial_trials; i++) {
        long tmp;
        conjugate_gradients_serial(matrix, rhs, sol, size, max_iters, rel_error, &tmp);
        serial_execution_time += tmp;

    }
    for(int i = 0; i < parallel_trials; i++) {
        long tmp;
        conjugate_gradients(matrix, rhs, sol, size, max_iters, rel_error, &tmp);
        parallel_execution_time += tmp;
    }

    double* y = new double[size];
    for(int i = 0; i < size; i++) {
        y[i] = 0;
    }
    gemv(1,matrix,sol,0,y,size,size);
    double err = 0;
    for(int i = 0; i < size; i++) {

        err += (rhs[i] - y[i]) * (rhs[i] - y[i]);
        if(std::isnan(sol[i])) {
            std::cout << "problem" << std::endl;
        }
    }
    std::cout << "residual norm is " << err << std::endl;

    std::cout << "check" << std::endl;
    check_cuda("error");
    std::cout << "Serial average execution time: " << (double)serial_execution_time/serial_trials << std::endl;
    std::cout << "Parallel average execution time: " << (double)parallel_execution_time/parallel_trials << std::endl;
    std::cout << "Speedup: " << (double)((double)serial_execution_time/serial_trials)/((double)parallel_execution_time/parallel_trials) << std::endl;
    printf("Finished successfully\n");


}
