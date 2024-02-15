#include <conjugate_gradients_cpu_openmp.hpp>

#include <cmath>
#include <cstdio>
#include <omp.h>

double omp_dot(const double * x, const double * y, size_t size)
{
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}

void omp_axpby(double alpha, const double * x, double beta, double * y, size_t size)
{
    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}



void omp_gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols)
{
    #pragma omp parallel for
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



void conjugate_gradients_cpu_openmp(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
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

    bb = omp_dot(b, b, size);
    rr = bb;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        omp_gemv(1.0, A, p, 0.0, Ap, size, size);
        alpha = rr / omp_dot(p, Ap, size);
        omp_axpby(alpha, p, 1.0, x, size);
        omp_axpby(-alpha, Ap, 1.0, r, size);
        rr_new = omp_dot(r, r, size);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / bb) < rel_error) { break; }
        omp_axpby(1.0, r, beta, p, size);
    }

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


