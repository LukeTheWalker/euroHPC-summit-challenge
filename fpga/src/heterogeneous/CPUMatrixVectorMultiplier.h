

#ifndef MATRIX_VECTOR_MULTIPLICATION_CPUMATRIXVECTORMULTIPLIER_H
#define MATRIX_VECTOR_MULTIPLICATION_CPUMATRIXVECTORMULTIPLIER_H

#include "MatrixVectorMultiplier.h"
class CPUMatrixVectorMultiplier : public MatrixVectorMultiplier{
public:
    virtual void init() {

    };

    virtual void setup() {

    };

    virtual void compute(double* p, double* Ap) {

#pragma omp parallel for simd num_threads(100) default(none) shared(Ap, p)
        for (size_t i = 0; i < partial_size; i += 1) {
            Ap[i] = 0.0;
#pragma omp simd
            for (size_t j = 0; j < size; j++) {
                Ap[i] += A[i * size + j] * p[j];
            }
        }
    };

    //size is the size of the size x size matrix
    virtual void setSize(size_t size_) {
        size = size_;
    };

    //size_ is the number of rows that are offloaded to the accelerator
    virtual void setPartialSize(size_t size_) {
        partial_size = size_;
    };


    virtual void setMatrix(double* matrix_) {
        A = matrix_;
    };

    virtual ~CPUMatrixVectorMultiplier() = default;

private:

    double* A;
    size_t size;
    size_t partial_size;

};


#endif //MATRIX_VECTOR_MULTIPLICATION_CPUMATRIXVECTORMULTIPLIER_H
