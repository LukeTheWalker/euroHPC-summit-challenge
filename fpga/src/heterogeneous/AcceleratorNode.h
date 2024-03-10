//
// Created by tomma on 04/03/2024.
//

#ifndef MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H
#define MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H
#include "utils.h"
#include "mpi.h"
template<typename Accelerator>
class AcceleratorNode {
public:
    void init() {
        accelerator.init();
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
        MPI_Recv(matrix, size * matrixData.partial_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        accelerator.setSize(size);
        accelerator.setPartialSize(matrixData.partial_size);
        accelerator.setMatrix(matrix);
        accelerator.setup();
    }

    void compute() {
        double* p = new (std::align_val_t(mem_alignment))double[size];
        double* Ap = new (std::align_val_t(mem_alignment))double[matrixData.partial_size];
        while(true) {

            MPI_Bcast(p, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            accelerator.compute(p, Ap);
            MPI_Gatherv(Ap, matrixData.partial_size, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        }
    }

    ~AcceleratorNode() {
        delete[] matrix;
    }

private:
    Accelerator accelerator;

    double* matrix;
    size_t max_memory = 2e30 * 16;
    size_t size;
    MPI_Datatype matrixDataType;
    size_t mem_alignment = 64;
    MatrixData matrixData;
    int rank;
};


#endif //MATRIX_VECTOR_MULTIPLICATION_ACCELERATORNODE_H
