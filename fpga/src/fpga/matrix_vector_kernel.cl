#define UNROLL 4
#define LATENCY 80
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




__kernel void matrix_vector_kernel(__global const double * __restrict__ A, __global const double * __restrict__ p, __global double * __restrict__ Ap, unsigned nrows, unsigned ncols )
{

#pragma unroll 4
    for(unsigned row = 0; row < nrows; row++) {
        Ap[row] = reduce(&A[row*ncols], p, ncols);
    }

}