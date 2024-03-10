cmake_minimum_required(VERSION 3.20)
project(matrix_vector_multiplication CUDA CXX)

include_directories(include)
include_directories(src/heterogeneous)
include_directories(include/CL)

add_executable(main src/conjugate_gradient.cpp )
add_executable(single_fpga src/fpga/single_fpga.cpp src/heterogeneous/utils.cpp)
add_executable(multi_fpga src/fpga/multi_fpga.cpp)
add_executable(multi_node src/heterogeneous/main.cpp src/heterogeneous/utils.cpp)
add_executable(mpi src/cpu/main_mpi.cpp src/heterogeneous/utils.cpp)
find_package(MPI REQUIRED)
target_link_libraries(multi_node MPI::MPI_CXX)
target_link_libraries(mpi MPI::MPI_CXX)


set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lOpenCL")


