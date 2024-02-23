#include <utils.cuh>
#include <unistd.h>

bool read_matrix_from_file(const char * filename, double ** matrix_out, size_t * num_rows_out, size_t * num_cols_out)
{
    double * matrix;
    size_t num_rows;
    size_t num_cols;

    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    #if USE_CUDA == 1
    cudaError_t err;
    err = cudaHostAlloc((void**)&matrix, num_rows * num_cols * sizeof(double), cudaHostAllocWriteCombined); cuda_err_check(err, __FILE__, __LINE__);
    #else
    matrix = new double[num_rows * num_cols];
    #endif
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

    return true;
}



bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols)
{
    FILE * file = fopen(filename, "wb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fwrite(&num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    fclose(file);

    return true;
}



void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file)
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}

void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
        exit (EXIT_FAILURE);
    }
}

void nccl_err_check (ncclResult_t err, const char *file, int line)
 {
     if (err != ncclSuccess)
     {
         fprintf (stderr, "NCCL error: %s (%s:%d)\n", ncclGetErrorString (err), file, line);
         exit (EXIT_FAILURE);
     }
 }

void mpi_err_check (int err, const char *file, int line)
{
    if (err != MPI_SUCCESS)
    {
        char err_string[MPI_MAX_ERROR_STRING];
        int err_string_len;
        MPI_Error_string (err, err_string, &err_string_len);
        fprintf (stderr, "MPI error: %s (%s:%d)\n", err_string, file, line);
        exit (EXIT_FAILURE);
    }
}

uint64_t getHostHash(const char *string)
{
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++)
    {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

void getHostName(char *hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++)
    {
        if (hostname[i] == '.')
        {
            hostname[i] = '\0';
            return;
        }
    }
}

