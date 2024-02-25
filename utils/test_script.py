import argparse
import subprocess
import struct
import os

reference_time = 43616498; # in milliseconds luca's machine

implementation_numbers = {
    'NCCL': 0,
    'MGPU': 0,
    'CUBLAS': 1,
    'TILED': 2,
    'ROWS': 3,
    'OPENMP': 4,
    'SERIAL': 5
}

def compile(version):
    flags = []
    if version != 'NCCL':
        flags += ['USE_NCCL=0']
    if version == 'OPENMP' or version == 'SERIAL':
        flags += ['USE_CUDA=0']

    # make clean
    clean_command = ['make', 'clean']
    with open(os.devnull, 'w') as devnull:
        subprocess.run(clean_command, check=True, stdout=devnull, stderr=devnull)

    # make with flags
    make_command = ['make', '-j4'] + flags
    # print the command being run
    with open(os.devnull, 'w') as devnull:
        subprocess.run(make_command, check=True, stdout=devnull, stderr=devnull)

def run(version, matrix_file, vector_file, output_file, tolerance, max_iterations):
    # make run with arguments
    run_command = ['./bin/conj', matrix_file, vector_file, output_file, tolerance, max_iterations, str(implementation_numbers[version])]
    subprocess.run(run_command, check=True)

def check_results(output_file, reference_file):
    # read the file written with:
    # fwrite(&num_rows, sizeof(size_t), 1, file);
    # fwrite(&num_cols, sizeof(size_t), 1, file);
    # fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    with open(output_file, 'rb') as file:
        num_rows = int.from_bytes(file.read(8), 'little')
        num_cols = int.from_bytes(file.read(8), 'little')
        output_matrix = struct.unpack('d' * (num_rows * num_cols), file.read(num_rows * num_cols * 8))

    # read the reference file
    with open(reference_file, 'rb') as file:
        num_rows = int.from_bytes(file.read(8), 'little')
        num_cols = int.from_bytes(file.read(8), 'little')
        reference_matrix = struct.unpack('d' * (num_rows * num_cols), file.read(num_rows * num_cols * 8))

    is_correct = True

    # compare the two matrices
    for i in range(len(output_matrix)):
        if abs(output_matrix[i] - reference_matrix[i]) > 1e-6:
            is_correct = False
            print(output_matrix[i], "vs",  reference_matrix[i])
    if is_correct:
        print('The output solution is correct')
    else:
        print('The output solution is incorrect')

def calculate_speedup(time_file, reference_time):
    # read the time file
    with open(time_file, 'r') as file:
        time = float(file.read())

    speedup = reference_time / time
    print('Speedup: ', speedup)

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Compile and run specific CG solver')

    # Add the version argument "cuBLAS", "MGPU", "Luca GPU", "Tommy GPU", "CPU (OpenMP)", "CPU (Serial)"
    parser.add_argument('implementation', choices=['NCCL', 'MGPU', 'CUBLAS', 'TILED', 'ROWS', 'OPENMP', 'SERIAL', 'ALL'], help='The version of the program to run')
    parser.add_argument('matrix_file', help='The file containing the matrix to solve')
    parser.add_argument('vector_file', help='The file containing the vector to solve')
    parser.add_argument('output_file', help='The file to write the solution to')
    parser.add_argument('tolerance', help='The tolerance for the solution')
    parser.add_argument('max_iterations', help='The maximum number of iterations to perform')
    parser.add_argument('reference_file', help='The reference file to compare the output to')

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.implementation == 'ALL':
        for version in implementation_numbers:
            print(f"---- Running {version} ----")
            compile(version)
            run(version, args.matrix_file, args.vector_file, args.output_file, args.tolerance, args.max_iterations)
            check_results(args.output_file, args.reference_file)
            calculate_speedup("output/time.txt", reference_time)
    else:
        compile(args.implementation)
        run(args.implementation, args.matrix_file, args.vector_file, args.output_file, args.tolerance, args.max_iterations)
        check_results(args.output_file, args.reference_file)
        calculate_speedup("output/time.txt", reference_time)
