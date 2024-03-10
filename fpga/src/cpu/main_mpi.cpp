#include "../src/cpu/MainNode.h"
#include "../src/cpu/AcceleratorNode.h"
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
    MPI_Init(nullptr, nullptr);
    int rank;
    int world_size;
    if(argc != 7) {
        std::cout << "wrong number of parameters" << std::endl;
        return 0;
    }
    int threads_number = atoi(argv[6]);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    long execution_time_fpga;

    if(rank == 0) {

        double* matrix;
        double* rhs;
        size_t tmp;
        int max_iter = atoi(argv[4]);
        double tol = atof(argv[5]);

        std::string matrix_path = argv[1];
        std::string rhs_path = argv[2];
        std::string output_path = argv[3];

        MainNode mainNode(matrix_path,
                                                      rhs_path,
                                                      output_path, max_iter, tol, threads_number);

        mainNode.init();

        auto start = std::chrono::high_resolution_clock::now();
        mainNode.handshake();
        mainNode.compute_conjugate_gradient();
        auto stop = std::chrono::high_resolution_clock::now();
        execution_time_fpga = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

        std::cout << "execution time = " << execution_time_fpga << std::endl;
        MPI_Finalize();
    } else {
        AcceleratorNode acceleratorNode(threads_number);
        acceleratorNode.init();
        acceleratorNode.handshake();
        acceleratorNode.compute();
        MPI_Finalize();
    }
}
