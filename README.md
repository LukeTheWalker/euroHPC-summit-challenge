# CG Solver

This is a program that solves a system of linear equations using the Conjugate Gradient (CG) method. It supports various implementations and matrix sizes.

## Usage

1. Navigate to the project directory:

    ```shell
    cd your_repository
    ```
2. Load modules:
    1. For FPGA and CPU:
        ```shell
        module load ifpgasdk
        module load 520nmx
        module load CMake
        module load intel
        module load deploy/EasyBuild
        module load Python
        ```
    2. For GPU:
        ```shell
        module load CUDA
        module load NCCL
        module load OpenMPI
        module load UCX
        module load UCX-CUDA
        module load Python
        ```

3. Run the program with the following command:

    ```shell
    python utils/test_script.py data_folder tolerance max_iterations implementation
    ```

    - `data_folder`: The folder containing the data files.
    - `tolerance`: The tolerance for the solution.
    - `max_iterations`: The maximum number of iterations to perform.
    - `implementation`: The version of the program to run. Choose from 'NCCL', 'MGPU', 'CUBLAS', 'TILED', 'ROWS', 'OPENMP', 'MPI', 'SERIAL', 'FPGA', 'MFPGA', 'MNFPGA', 'ALL', 'ALL_CPU', 'ALL_FPGA'.
