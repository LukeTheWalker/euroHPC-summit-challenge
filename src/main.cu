#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <conjugate_gradients_gpu.cu>
#include <conjugate_gradients_cpu_serial.hpp>
#include <conjugate_gradients_cpu_openmp.hpp>
#include <conjugate_gradients_gpu_tommy.cu>
#include <utils.hpp>
#include <functional>
#include <string>

int main(int argc, char ** argv)
{
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

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

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("\n");

    double * matrix;
    double * rhs;
    size_t size;

    {
        printf("Reading matrix from file ...\n");
        size_t matrix_rows;
        size_t matrix_cols;
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
        if(!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }
        printf("Done\n");
        printf("\n");

        printf("Reading right hand side from file ...\n");
        size_t rhs_rows;
        size_t rhs_cols;
        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
        if(!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        printf("Done\n");
        printf("\n");

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

    int number_of_tests = 4;
    int times[number_of_tests];
    double sol [number_of_tests][size];
    std::function<void(double*, double*, double*, size_t, int, double)> implementations_to_test[number_of_tests] = 
    {luca::par_conjugate_gradients, tommy::conjugate_gradients<true>, conjugate_gradients_cpu_openmp, conjugate_gradients_cpu_serial};
    std::string names[number_of_tests] = {"Luca GPU", "Tommy GPU", "CPU (OpenMP)", "CPU (Serial)"};
    int order[number_of_tests] = {0, 1, 2, 3};
    
    for (auto i : order)
    {
        cudaDeviceReset();
        times[i] = 0;
        printf("Solving the system with %s ...\n", names[i].c_str());
        double start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        implementations_to_test[i](matrix, rhs, sol[i], size, max_iters, rel_error);
        double end_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        times[i] = (end_time - start_time);
        printf("Done in %f milliseconds\n", times[i] / 1000.0);
        printf("\n");
    }

    // calculate speedup
    for (int i = 0; i < number_of_tests - 1; i++)
    {
        printf("Speedup of function %s: %f\n", names[i].c_str(), (double)times[number_of_tests - 1] / times[i]);
    }

    printf("Writing solution to file ...\n");
    bool success_write_sol = write_matrix_to_file(output_file_sol, sol[3], size, 1);
    if(!success_write_sol)
    {
        fprintf(stderr, "Failed to save solution\n");
        return 6;
    }
    printf("Done\n");
    printf("\n");

    // check if all solutions are the same as the serial one, otherwise print the first 10 elements of each solution and the error
    bool all_same = true;
    for(size_t i = 0; i < 4; i++)
    {
        for(size_t j = 0; j < size; j++)
        {
            if(fabs(sol[3][j] - sol[i][j]) > 1e-6)
            {
                printf("Solution %ld is different from the serial one\n", i);
                printf("First 10 elements of the serial solution:\n");
                for(int k = 0; k < 10; k++)
                {
                    printf("%f\n", sol[3][k]);
                }
                printf("\n");
                printf("First 10 elements of the solution %ld:\n", i);
                for(int k = 0; k < 10; k++)
                {
                    printf("%f\n", sol[i][k]);
                }
                printf("\n");
                printf("Error: %e\n", fabs(sol[3][j] - sol[i][j]));
                printf("\n");
                break;
            }
        }
        if(!all_same) break;
    }
   

    delete[] matrix;
    delete[] rhs;
    

    printf("Finished successfully\n");

    return 0;
}
