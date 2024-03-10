#include "MainNode.h"
#include "AcceleratorNode.h"
#include "CPUMatrixVectorMultiplier.h"
#include <chrono>



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
    std::cout << "mpi version" << std::endl;
    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    long execution_time_fpga;

    if(rank == 0) {

        double* matrix;
        double* rhs;
        size_t size;
        size_t tmp;
        int max_iter = atoi(argv[4]);
        double tol = atof(argv[5]);

        std::string matrix_path = argv[1];
        std::string rhs_path = argv[2];
        std::string output_path = argv[3];

        MainNode<CPUMatrixVectorMultiplier> mainNode(matrix_path,
                                                      rhs_path,
                                                      output_path, max_iter, tol);

        mainNode.init();

        auto start_fpga = std::chrono::high_resolution_clock::now();
        mainNode.handshake();
        mainNode.compute_conjugate_gradient();
        auto stop_fpga = std::chrono::high_resolution_clock::now();
        execution_time_fpga = std::chrono::duration_cast<std::chrono::microseconds>(stop_fpga - start_fpga).count();

        std::cout << "execution time = " << execution_time_fpga << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    } else {
        AcceleratorNode<CPUMatrixVectorMultiplier> acceleratorNode;
        acceleratorNode.init();
        acceleratorNode.handshake();
        acceleratorNode.compute();
    }
}
