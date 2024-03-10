#ifndef MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H
#define MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H
#include "utils.h"
#include "mpi.h"
class AcceleratorNode {
public:
    AcceleratorNode(int threads_number) : threads_number(threads_number){}
    void init() {
    }

    void handshake() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        size_t max_rows = max_memory / (size * sizeof (double));
        MPI_Gather(&max_rows, 1, MPI_UNSIGNED_LONG, &max_rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Type_contiguous(2, MPI_UNSIGNED_LONG, &matrixDataType);
        MPI_Type_commit(&matrixDataType);
        MPI_Scatter(NULL, 0, matrixDataType, &matrixData, 1, matrixDataType, 0, MPI_COMM_WORLD);
        matrix = new double[size * matrixData.partial_size];
        #pragma omp parallel for default(none) num_threads(threads_number)
        for(int i = 0; i < size * matrixData.partial_size; i++) {
            matrix[i] = 0.0;
        }

        MPI_Recv(matrix, size * matrixData.partial_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


    }

    void compute() {

        double* p = new (std::align_val_t(mem_alignment))double[size];
        double* Ap = new (std::align_val_t(mem_alignment))double[matrixData.partial_size];
        #pragma omp parallel for default(none) shared(p) num_threads(threads_number)
        for(int i = 0; i < size;i++) {
            p[i] = 0;
        }
        #pragma omp parallel for default(none) shared(Ap) num_threads(threads_number)
        for(int i = 0; i < matrixData.partial_size;i++) {
            Ap[i] = 0;
        }
        bool finished;

        #pragma omp parallel default(none) shared(finished, p, Ap, matrixData) num_threads(threads_number)
        {
            while (true) {


                #pragma omp single
                {
                    MPI_Bcast(&finished, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
                }
                if(finished) {
                    break;
                }
                #pragma omp single
                {
                    MPI_Request r;
                    MPI_Ibcast(p, size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &r);
                    MPI_Wait(&r, MPI_STATUS_IGNORE);
                }


                #pragma omp for simd
                for (size_t i = 0; i < matrixData.partial_size; i += 1) {
                    Ap[i] = 0.0;
                    #pragma omp simd
                    for (size_t j = 0; j < size; j++) {
                        Ap[i] += matrix[i * size + j] * p[j];
                    }
                }
                #pragma omp single
                {
                    MPI_Gatherv(Ap, matrixData.partial_size, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0,
                                MPI_COMM_WORLD);

                }


            }
        }

    }

    ~AcceleratorNode() {
        delete[] matrix;
    }

private:

    double* matrix;
    size_t max_memory = 2e30 * 512;
    size_t size;
    MPI_Datatype matrixDataType;
    size_t mem_alignment = 64;
    MatrixData matrixData;
    int rank;
    int threads_number;
};


#endif //MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H