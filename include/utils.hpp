#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cstdio>

bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out);
bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols);
void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file = stdout);

#endif // UTILS_HPP