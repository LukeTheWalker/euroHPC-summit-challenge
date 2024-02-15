#ifndef CONJUGATE_GRADIENTS_CPU_OPENMP_HPP
#define CONJUGATE_GRADIENTS_CPU_OPENMP_HPP

#include <cstddef>

void conjugate_gradients_cpu_openmp(const double * A, const double * b, double * x, size_t size, int max_iters, double rel_error);

#endif // CONJUGATE_GRADIENTS_CPU_HPP