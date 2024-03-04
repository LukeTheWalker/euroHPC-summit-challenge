#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <bits/stdc++.h>

double dot (const double * x, const double * y, size_t size) {
    double result = 0.0;
    #pragma omp simd reduction(+:result)
    for(size_t i = 0; i < size; i++) {
        result += x[i] * y[i];
    }
    return result;
}


void conjugate_gradients_cpu_openmp(const double *  A, const double * b, double * x, size_t size, int max_iters, double rel_error) {
    double alpha = 0, beta = 0, bb = 0, rr = 0, rr_new = 0;
    auto * r = new double[size];
    auto * p = new double[size];
    auto * Ap = new double[size];
    int num_iters = 0;

    for(size_t i = 0; i < size; i++) {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, size);
    rr = bb;
    double dot_result = 0.0;
    rr_new = 0.0;
    int total_iterations = 0;
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel default(none) shared(max_iters, size, rel_error, A, p, Ap, x, r, dot_result, rr_new, total_iterations) firstprivate(alpha, beta, rr, bb, num_iters)
    {
        for (num_iters = 1; num_iters <= max_iters; num_iters++) {

            #pragma omp for simd nowait
            for (size_t i = 0; i < size; i += 1) {
                Ap[i] = 0.0;
                #pragma omp simd
                for (size_t j = 0; j < size; j++) {
                    Ap[i] += A[i * size + j] * p[j];
                }
            }



            #pragma omp single
            {
                dot_result = 0.0;
                rr_new = 0.0;
            }


            #pragma omp for simd reduction(+:dot_result)
            for (size_t i = 0; i < size; i++) {
                dot_result += p[i] * Ap[i];
            }
            alpha = rr / dot_result;


            #pragma omp for simd nowait
            for(size_t i = 0; i < size; i++) {
                x[i] = alpha * p[i] + x[i];
            }


            #pragma omp for simd nowait
            for(size_t i = 0; i < size; i++) {
                r[i] = -alpha * Ap[i] + r[i];
            }


            #pragma omp for simd reduction(+:rr_new)
            for (size_t i = 0; i < size; i++) {
                rr_new += r[i] * r[i];
            }


            beta = rr_new / rr;
            rr = rr_new;
            if (std::sqrt(rr / bb) < rel_error) {
                #pragma omp single
                {
                    total_iterations = num_iters;
                }
                break; }

            #pragma omp for simd
            for(size_t i = 0; i < size; i++) {
                p[i] =  r[i] + beta * p[i];
            }
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    printf("Net OpenMP time: %fms\n", std::chrono::duration<double, std::milli>(stop - start).count());

    delete[] r;
    delete[] p;
    delete[] Ap;
    if(total_iterations <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", total_iterations, std::sqrt(rr_new / bb));
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", total_iterations, std::sqrt(rr_new / bb));
    }
}