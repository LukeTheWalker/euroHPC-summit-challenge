//
// Created by tomma on 02/03/2024.
//

#ifndef MATRIX_VECTOR_MULTIPLICATION_MAINNODE_H
#define MATRIX_VECTOR_MULTIPLICATION_MAINNODE_H

#include <string>
#include <fstream>
#include <utility>
#include <vector>
#include <mpi.h>
#include <math.h>
#include "utils.h"
#include <algorithm>


template<typename Accelerator>
class MainNode {
public:

    MainNode(std::string  matrix_file_path, std::string  rhs_file_path, int max_iters, double tol) : matrix_file_path(std::move(matrix_file_path)), rhs_file_path(std::move(rhs_file_path)), max_iters(max_iters), tol(tol) {}

    void init() {

        accelerator.init();
    }
    void handshake() {

        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        max_size.resize(world_size);
        //std::ifstream is;
        /*is.open(matrix_file_path, std::ios::binary);
        is.read((char*)&size,sizeof(size_t));
        is.close();
         */

        read_rhs();
        std::cout << "rhs read, size = "  << size << std::endl;
        MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        max_size[0] = max_memory / (size * sizeof(double));
        MPI_Gather(MPI_IN_PLACE, 1, MPI_UNSIGNED_LONG, &max_size[0], 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);


        size_t total_capacity = 0;
        for(int i = 0; i < world_size; i++) {
            total_capacity += max_size[i];
        }
        std::vector<double> quota(world_size);
        for(int i = 0; i < world_size; i++) {
            quota[i] = (double)max_size[i]/(double)total_capacity;
        }

        offset.resize(world_size);
        partial_size.resize(world_size);
        offset[0] = 0;
        for(int i = 1; i < world_size; i++) {
            offset[i] = offset[i-1] + std::min((size_t)(size*quota[i-1]), max_size[i-1]);
            partial_size[i-1] = std::min((size_t)(size*quota[i-1]), max_size[i-1]);
        }
        partial_size[world_size - 1] = size - offset[world_size - 1];

        MPI_Datatype matrixDataType;
        MPI_Type_contiguous(2, MPI_UNSIGNED_LONG, &matrixDataType);
        MPI_Type_commit(&matrixDataType);

        std::vector<MatrixData> matrixData(world_size);
        for(int i = 0; i < world_size; i++) {
            matrixData[i] = {(size_t)offset[i], (size_t)partial_size[i]};
        }

        MPI_Scatter(&matrixData[0], 1, matrixDataType, &myMatrixData, 1, matrixDataType, 0, MPI_COMM_WORLD);


        sol.resize(size);

        read_and_send_matrix();
        accelerator.setSize(size);
        accelerator.setPartialSize(matrixData[0].partial_size);
        accelerator.setMatrix(matrix);
        accelerator.setup();
    }

    void compute_conjugate_gradient() {
        //std::cout << "starting to compute" << std::endl;
        double alpha;
        double beta;
        double rr;
        double rr_new;
        double bb;
        std::vector<double> r(size);

        double* p = new (std::align_val_t(mem_alignment))double[size];
        double* Ap = new (std::align_val_t(mem_alignment))double[size];
        //std::cout << "check1" << std::endl;


        r = rhs;
        for(int i = 0; i < size; i++) {
            p[i] = rhs[i];
        }
        bb = dot(rhs,rhs,size);
        rr = bb;
        for(auto& s : sol) {
            s = 0.0;
        }
        int iters;
        for(iters = 1; iters <= max_iters; iters++) {
            //std::cout << "iteration " << iters << std::endl;
            //MPI_Request request1, request2;
            MPI_Bcast(&p[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &Ap[0], (&(partial_size[0])),
                        (&(offset[0])), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            accelerator.compute(p, Ap);
            //MPI_Wait(&request2, MPI_STATUS_IGNORE);
            alpha = rr / dot(p, Ap, size);
            axpby(alpha, p, 1.0, sol, size);
            axpby(-alpha, Ap, 1.0, r, size);
            rr_new = dot(r, r, size);
            beta = rr_new / rr;
            rr = rr_new;
            if(std::sqrt(rr / bb) < tol) { break; }
            axpby(1.0, r, beta, p, size);
        }

        if(iters <= max_iters)
        {
            printf("Converged in %d iterations, relative error is %e\n", iters, std::sqrt(rr / bb));
        }
        else
        {
            printf("Did not converge in %d iterations, relative error is %e\n", iters, std::sqrt(rr / bb));
        }

        delete[] Ap;
        delete[] p;

        //MPI_Abort(MPI_COMM_WORLD, 0);
    }

    ~MainNode() {
        delete[] matrix;
    }
private:


    void read_rhs() {
        std::ifstream is;
        size_t tmp;
        is.open(rhs_file_path, std::ios::binary);
        is.read((char*)&size,sizeof(size_t));
        is.read((char*)&tmp,sizeof(size_t));
        rhs.resize(size);
        is.read((char*)&rhs[0], size * sizeof(double));
        is.close();
    }

    /*
    void read_rhs() {
        rhs.resize(size);
        for(auto& r : rhs) {
            r = 1.0;
        }
    }
     */


/*
    void read_and_send_matrix() {
        std::vector<double> matrix_(size * size);
        for(size_t i = 0; i < size * size; i++) {
            matrix_[i] = 0.0;
        }
        for(size_t i = 0; i < size; i++) {
            matrix_[i * size + i] = 2.0;
            if(i != size-1) {
                matrix_[(i + 1) * size + i] = -1;
                matrix_[i * size + (i + 1)] = -1;
            }
        }

        matrix = new (std::align_val_t(mem_alignment)) double[size * myMatrixData.partial_size];
        for(size_t i = 1; i < world_size; i++) {
            MPI_Send(&matrix_[0] + offset[i] * size, size * partial_size[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        for(int i = 0; i < size * myMatrixData.partial_size; i++) {
            matrix[i] = matrix_[i];
        }
    }
    */

    void check_matrix(double* matrix, int nrows, int offset) {
        std::vector<double> matrix_(size * size);
        for(size_t i = 0; i < size * size; i++) {
            matrix_[i] = 0.0;
        }
        for(size_t i = 0; i < size; i++) {
            matrix_[i * size + i] = 2.0;
            if(i != size-1) {
                matrix_[(i + 1) * size + i] = -1;
                matrix_[i * size + (i + 1)] = -1;
            }
        }
        for(int i = 0; i < size * nrows; i++) {
            if(matrix_[i + offset*size] != matrix[i]) {
                std::cout << "matrices doesnt match " << matrix_[i] << " " << matrix[i] << std::endl;
            }
        }
    }

    void read_and_send_matrix() {
        auto it = std::max_element(partial_size.begin(), partial_size.end());
        size_t msize = *it;
        double* matrix_ = new (std::align_val_t(mem_alignment)) double[msize * size];
        matrix = new (std::align_val_t(mem_alignment)) double[partial_size[0] * size];
        std::ifstream is;
        size_t buff;
        is.open(matrix_file_path, std::ios::binary);
        if(!is.is_open()) {
            std::cout << "matrix file doesn't exist" << std::endl;
            exit(1);
        }
        is.read((char*)&buff, sizeof(size_t));
        is.read((char*)&buff, sizeof(size_t));
        is.read((char*)matrix, size * partial_size[0] * sizeof(double));
        //check_matrix(matrix, partial_size[0], 0);

        for(int i = 1; i < world_size; i++) {
            is.read((char*)matrix_, size * partial_size[i] * sizeof(double));
            MPI_Send(matrix_, size * partial_size[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        is.close();
        delete[] matrix_;
        std::cout << "matrix read, size = " << size << std::endl;

    }

    std::string matrix_file_path;
    std::string rhs_file_path;
    size_t size;
    int world_size;
    std::vector<int> offset;
    std::vector<int> partial_size;
    std::vector<size_t> max_size;
    std::vector<double> rhs;
    std::vector<double> sol;
    double* matrix;
    size_t total_device_number;
    int max_iters;
    double tol;
    MatrixData myMatrixData;
    int mem_alignment = 64;
    size_t max_memory = 2e30 * 16;


    Accelerator accelerator;

};


#endif //MATRIX_VECTOR_MULTIPLICATION_MAINNODE_H
