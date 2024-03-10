
#define UNROLL 3
#define LATENCY 64
//#define SIZE 10000

double reduce(__global const double * __restrict__ array1, __global const double * __restrict__ array2, unsigned size) {
    double shift_reg[LATENCY] = {0.0};
    double final_sum = 0.0;

    unsigned exit = (size % UNROLL == 0)?(size/UNROLL):((size/UNROLL) + 1);
    for(unsigned i = 0; i < exit; i++) {
        double sum = 0.0;
#pragma unroll
        for(unsigned j = 0; j < UNROLL; j++) {
            unsigned  index = i * UNROLL + j;
            sum += (index < size) ? array1[index]*array2[index] : 0.0;
        }
        double old_shift_reg_0 = shift_reg[0];
#pragma unroll
        for(unsigned j = 0; j < LATENCY - 1; j++) {
            shift_reg[j] = shift_reg[j + 1];
        }
        shift_reg[LATENCY - 1] = old_shift_reg_0 + sum;
    }
#pragma unroll
    for(unsigned i = 0; i < LATENCY; i++) {
        final_sum += shift_reg[i];
    }

    return final_sum;
}




__kernel void conjugate_gradient_kernel(__global const double * __restrict__ A, __global const double * __restrict__ b, __global double * __restrict__ x, unsigned size, __global double * __restrict__ r, __global double * __restrict__ p, __global double* __restrict__ Ap, unsigned max_iters, double squared_tol, double bb, __global int* actual_iterations, __global double* actual_error)
{
    double alpha, beta, rr, rr_new;
    int num_iters;

    rr = bb;



    for(num_iters = 1; num_iters <= max_iters; num_iters++) {

        //matrix vector multiplication
#pragma unroll 4
        for(unsigned row = 0; row < size; row++) {
            Ap[row] = reduce(&A[row*size], p, size);
        }

        double tmp_dot_result = reduce(Ap, p, size);
        alpha = rr/tmp_dot_result;

//#pragma unroll UNROLL
        for(unsigned i = 0; i < size; i++) {
            x[i] += alpha * p[i];
        }

//#pragma unroll UNROLL
        for(unsigned i = 0; i < size; i++) {
            r[i] += -alpha * Ap[i];
        }

        rr_new = reduce(r,r,size);

        beta = rr_new /  rr;
        rr = rr_new;
        if(rr/bb < squared_tol) {break;}

//#pragma unroll UNROLL
        for(unsigned i = 0; i < size; i++) {
            p[i] = r[i] + beta*p[i];
        }
    }

    *actual_iterations = num_iters;
    *actual_error = rr/bb;
}