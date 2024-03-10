//
// Created by tomma on 04/03/2024.
//

#include <stdlib.h>
#include <iostream>
#include <fstream>

double* matrix_generator(int size) {
    double* matrix_ = new double[size * size];
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
    return matrix_;
}


int main(int argc, char ** argv) {
    int size = atoi(argv[1]);
    double* matrix = matrix_generator(size);
    std::cout << "Generating a " << size << " x " << size << " matrix";
    std::ofstream fs("matrix.bin", std::ios::out | std::ios::binary);
    fs.write((char*)&size, sizeof(int));
    fs.write((char*)matrix, size * size * sizeof(double));
    std::cout << "Completed generation of the matrix" << std::endl;
}