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

#if USE_CUDA == 1
void cuda_err_check (cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf (stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString (err), file, line);
        exit (EXIT_FAILURE);
    }
}
#endif

#if USE_NCCL == 1
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

ncclUniqueId id;
ncclComm_t * comms;
int myRank, nRanks, localRank = 0;

void initialize_nccl () {

    int mpi_err; ncclResult_t nccl_err; cudaError_t cuda_err;

    // get localRank
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, (int*)&myRank); mpi_err_check(mpi_err, __FILE__, __LINE__);
    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, (int*)&nRanks); mpi_err_check(mpi_err, __FILE__, __LINE__);

    int nDevices;
    cuda_err = cudaGetDeviceCount(&nDevices); cuda_err_check(cuda_err, __FILE__, __LINE__);

    comms = (ncclComm_t *)malloc(nDevices * sizeof(ncclComm_t));

    // calculating localRank which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    mpi_err = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD); mpi_err_check(mpi_err, __FILE__, __LINE__);
    for (int p = 0; p < nRanks; p++)
    {
        if (p == myRank)
            break;
        if (hostHashs[p] == hostHashs[myRank])
            localRank++;
    }

    if (myRank == 0)
        ncclGetUniqueId(&id);
    mpi_err = MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD); mpi_err_check(mpi_err, __FILE__, __LINE__);

    nccl_err = ncclGroupStart(); nccl_err_check(nccl_err, __FILE__, __LINE__);
    for (int i = 0; i < nDevices; i++)
    {
        cuda_err = cudaSetDevice(localRank * nDevices + i); cuda_err_check(cuda_err, __FILE__, __LINE__);
        nccl_err = ncclCommInitRank(comms + i, nRanks * nDevices, id, myRank * nDevices + i); nccl_err_check(nccl_err, __FILE__, __LINE__);
    }
    nccl_err = ncclGroupEnd(); nccl_err_check(nccl_err, __FILE__, __LINE__);
    fprintf(stderr,"[MPI Rank %d] responsible for GPU %d-%d\n", myRank, myRank * nDevices, myRank * nDevices + nDevices - 1);
}
#endif