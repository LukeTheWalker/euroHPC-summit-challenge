

#ifndef MATRIX_VECTOR_MULTIPLICATION_MATRIXVECTORMULTIPLIER_H
#define MATRIX_VECTOR_MULTIPLICATION_MATRIXVECTORMULTIPLIER_H

#include <stdlib.h>

class MatrixVectorMultiplier {
public:
    virtual void init() = 0;

    virtual void setup() = 0;

    virtual void compute(double* p, double* Ap) = 0;

    //size is the size of the size x size matrix
    virtual void setSize(size_t size_) = 0;

    //size_ is the number of rows that are offloaded to the accelerator
    virtual void setPartialSize(size_t size_) = 0;


    virtual void setMatrix(double* matrix_) = 0;

    virtual ~MatrixVectorMultiplier() = default;

};


#endif //MATRIX_VECTOR_MULTIPLICATION_MATRIXVECTORMULTIPLIER_H