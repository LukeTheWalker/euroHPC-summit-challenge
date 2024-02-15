#ifndef CONJUGATE_GRADIENTS_CPU_HPP
#define CONJUGATE_GRADIENTS_CPU_HPP

#include <cstddef>

double dot(const double * x, const double * y, size_t size);
void axpby(double alpha, const double * x, double beta, double * y, size_t size);
void gemv(double alpha, const double * A, const double * x, double beta, double * y, size_t num_rows, size_t num_cols);
void conjugate_gradients(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error);

#endif // CONJUGATE_GRADIENTS_CPU_HPP