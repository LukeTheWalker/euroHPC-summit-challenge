__kernel void gemv(__global const double * a, __global const double * x, __global double * y, int ncols)
{
    for(int i = 0; i < ncols; i++) {
        y[i] = a[i] + y[i];
    }
}