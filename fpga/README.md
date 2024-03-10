# OpenMP, MPI + openMP and  FPGA versions

This folder contains the OpenMP, MPI+OpenMP and FPGA implementatiions of the Conjugate Gradient algorithm

## Compilation
In order to compile the code, you first need to create a subdirectory.\
```
mkdir build
cd build
```
\
Then, it is necessary to load some modules:
```module load ifpgasdk
module load 520nmx
module load CMake
module load intel
module load deploy/EasyBuild
```
Then
```
cmake ..
```
And finally,
```
make
```
At this point you can run the code.

For FPGA versions it is necessary to download the needed kernels.
They can be found at this **[link](https://drive.google.com/file/d/1vZyDhI-ukpShtWAkAV_XjTLTUKVczaEf/view?usp=sharing)**
When downloaded, they should be pasted in ths /src/fpga folder.
## OpenMP
In order to run the OpenMP version
```
./main max_iters rel_err number_of_serial_runs number_of_parallel_runs threads_number matrix_path rhs_path output_path
```
After running, the program outputs the average parallel and serial execution time, alongside the achieved speed up.
In addition, the solution of the system is written to the output file
## FPGA
For the FPGA, there are 3 different versions:
### Single FPGA
To run the single FPGA version
```
./single_fpga matrix_path rhs_path max_iters rel_error
```
The program outputs the parallel execution time.
In addition, the solution og the system is written to the output file.
## Multi FPGA
To run the multi FPGA version
```
./multi_fpga matrix_path rhs_path max_iters rel_error num_devices
```
The program ouputs both the parallel and the serial execution time.
In addition, the solution of the system is written to the output file.

## Multi Nodes FPGA
To run this version on Meluxina, after reserving an appropriate number of nodes
```
srun multi_node matrix_path rhs_path output_path max_iters rel_err threads_number
```
The program outputs the execution time.
In addition, the solution of the system is written to the output file.

## MPI + OpenMP
To run this version on Meluxina, after reserving an appropriate number of nodes
```
srun mpi matrix_path rhs_path output_path max_iters rel_err threads_number
```

