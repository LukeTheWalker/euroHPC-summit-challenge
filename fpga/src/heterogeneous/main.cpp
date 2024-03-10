//
// Created by tomma on 03/03/2024.
//

#include "MainNode.h"
#include "FPGAMatrixVectorMultiplier.h"
#include "AcceleratorNode.h"
//#include "GPUMatrixVectorMultiplier.cuh"
#include <chrono>



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



void conjugate_gradients_parallel(const double *  A, const double * b, double * x, size_t size, int max_iters, double rel_error, int threads_number, long* execution_time) {
    double alpha, beta, bb, rr, rr_new;
    auto * r = new double[size];
    auto * p = new double[size];
    auto * Ap = new double[size];
    int num_iters;

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
#pragma omp parallel default(none) shared(max_iters, size, rel_error, A, p, Ap, x, r, dot_result, rr_new, total_iterations) firstprivate(alpha, beta, rr, bb, num_iters) num_threads(threads_number)
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
    *execution_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

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






void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error)
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

bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}



int main(int argc, char** argv) {
    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    long execution_time_fpga, execution_time_serial;




    if(rank == 0) {
        double* matrix;
        double* rhs;
        size_t size;
        size_t tmp;
        int max_iter = atoi(argv[3]);
        double tol = atof(argv[4]);
        auto start_serial = std::chrono::high_resolution_clock::now();
        read_matrix_from_file(argv[1], &matrix, &size, &size);
        read_matrix_from_file(argv[2], &rhs, &tmp, &tmp);
        double* sol = new double[size];
        conjugate_gradients(matrix, rhs, sol, size, max_iter, tol);
        auto stop_serial = std::chrono::high_resolution_clock::now();
        execution_time_serial = std::chrono::duration_cast<std::chrono::microseconds>(stop_serial - start_serial).count();
        std::cout << "starting fpga version" << std::endl;
        MainNode<FPGAMatrixVectorMultiplier> mainNode(argv[1], argv[2], max_iter, tol);
        mainNode.init();
        auto start_fpga = std::chrono::high_resolution_clock::now();
        mainNode.handshake();
        mainNode.compute_conjugate_gradient();
        auto stop_fpga = std::chrono::high_resolution_clock::now();
        execution_time_fpga = std::chrono::duration_cast<std::chrono::microseconds>(stop_fpga - start_fpga).count();

        std::cout << "fpga execution time = " << execution_time_fpga << std::endl;
        std::cout << "serial execution time = " << execution_time_serial << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);


    } else {
        AcceleratorNode<FPGAMatrixVectorMultiplier> acceleratorNode;
        acceleratorNode.init();
        acceleratorNode.handshake();
        acceleratorNode.compute();
    }






}
