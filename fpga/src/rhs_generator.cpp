//
// Created by tomma on 04/03/2024.
//

#include <stdlib.h>
#include <iostream>
#include <fstream>
int main(int argc, char** argv) {
    int size = atoi(argv[1]);
    double* rhs = new double[size];
    for(int i = 0; i < size; i++) {
        rhs[i] = 1.0;
    }

    std::ofstream fs("rhs.bin", std::ios::out | std::ios::binary | std::ios::app);
    fs.write((char*)&size, sizeof(int));
    fs.write((char*)rhs, size * sizeof(double));
    std::cout << "Completed generation of the rhs" << std::endl;
}