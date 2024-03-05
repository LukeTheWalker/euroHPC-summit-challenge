#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#include <conjugate_gradients_cpu_serial.hpp>
#include <conjugate_gradients_cpu_openmp.hpp>
#include <utils.cuh>
#include <functional>
#include <string>

#if USE_CUDA == 1
#include <conjugate_gradients_gpu_tommy.cu>
#include <conjugate_gradients_multi_gpu.cu>
#include <conjugate_gradients_cublas.cu>
#include <conjugate_gradients_gpu.cu>
#endif


int main(int argc, char ** argv)
{
    // printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    // printf("All parameters are optional and have default values\n");
    // printf("\n");

    const char * input_file_matrix = "io/matrix.bin";
    const char * input_file_rhs = "io/rhs.bin";
    const char * output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 3) output_file_sol = argv[3];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);

    // printf("Command line arguments:\n");
    // printf("  input_file_matrix: %s\n", input_file_matrix);
    // printf("  input_file_rhs:    %s\n", input_file_rhs);
    // printf("  output_file_sol:   %s\n", output_file_sol);
    // printf("  max_iters:         %d\n", max_iters);
    // printf("  rel_error:         %e\n", rel_error);
    // printf("\n");

    double * matrix;
    double * rhs;
    size_t size;

    {
        // printf("Reading matrix from file ...\n");
        size_t matrix_rows;
        size_t matrix_cols;
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        // printf("Done\n");
        // printf("\n");

        // printf("Reading right hand side from file ...\n");
        size_t rhs_rows;
        size_t rhs_cols;
        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
        if(!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        // printf("Done\n");
        // printf("\n");

        if(matrix_rows != matrix_cols)
        {
            fprintf(stderr, "Matrix has to be square\n");
            return 3;
        }
        if(rhs_rows != matrix_rows)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return 4;
        }
        if(rhs_cols != 1)
        {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }

        size = matrix_rows;
    }

    int exec_time;
    double sol[size];
    std::function<void(double*, double*, double*, size_t, int, double)> implementations_to_test[255]; 
        
        #if USE_CUDA == 1
        implementations_to_test[0] = luca::par_conjugate_gradients_multi_gpu;
        implementations_to_test[1] = conjugate_gradients_cublas;
        implementations_to_test[2] = luca::par_conjugate_gradients;
        implementations_to_test[3] = tommy::conjugate_gradients; 
        #endif
        implementations_to_test[4] = conjugate_gradients_cpu_openmp; 
        implementations_to_test[5] = conjugate_gradients_cpu_serial;

    std::string names[] = {"MGPU", "cuBLAS", "Luca GPU", "Tommy GPU", "CPU (OpenMP)", "CPU (Serial)"};

    int impl_used = argc > 6 ? atoi(argv[6]) : 0;

    {
        #if USE_CUDA == 1
        int number_of_gpus; cudaError_t err;
        err = cudaGetDeviceCount(&number_of_gpus); cuda_err_check(err, __FILE__, __LINE__);
        for (int i = 0; i < number_of_gpus; i++)
        {
            err = cudaSetDevice(i); cuda_err_check(err, __FILE__, __LINE__);
            err = cudaFree(0); cuda_err_check(err, __FILE__, __LINE__);
        }
        #endif
        
        exec_time = 0;
        printf("Solving the system with %s ...\n", names[impl_used].c_str());
        double start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        implementations_to_test[impl_used](matrix, rhs, sol, size, max_iters, rel_error);
        double end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        exec_time = (end_time - start_time);
        printf("Done in %f milliseconds\n", exec_time);
    }

    
    // printf("Writing solution to file ...\n");
    bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
    if(!success_write_sol)
    {
        fprintf(stderr, "Failed to save solution\n");
        return 6;
    }

    FILE * time_f = fopen("output/time.txt", "w");
    fprintf(time_f, "%d", exec_time);
    fclose(time_f);

    #if USE_CUDA == 1
    cudaError_t err;
    err = cudaFreeHost(matrix); cuda_err_check(err, __FILE__, __LINE__);
    err = cudaFreeHost(rhs); cuda_err_check(err, __FILE__, __LINE__);
    #endif

    printf("Finished successfully\n");

    return 0;
}
