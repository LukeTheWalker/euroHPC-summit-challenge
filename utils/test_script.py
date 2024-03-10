import argparse
import subprocess
import struct
import os
import pandas as pd

# reference_time = 43616498; # in milliseconds luca's machine
reference_time = 58690450; # meluxina

implementation_numbers = {
    'NCCL': 0,
    'MGPU': 0,
    'CUBLAS': 1,
    'TILED': 2,
    'ROWS': 3,
    # 'OPENMP': 4,
    'SERIAL': 5
}

matrix_size = [
    100,
    500,
    1000,
    5000,
    10000,
    20000,
    30000,
    40000,
    50000,
    60000,
    70000
]

tommy_implementations = [
    'FPGA',
    'MFPGA',
    'MNFPGA',
    'OPENMP',
    'MPI'
]

executable_names = {
    'FPGA': 'single_fpga',
    'MFPGA': 'multi_fpga',
    'MNFPGA': 'multi_node',
    'OPENMP': 'main',
    'MPI': 'mpi'
}

threads_number = 200;


def compile(version):
    print(f'----------------- Using {version} -----------------')

    if version in tommy_implementations:
        #change directory to the FPGA directory
        os.chdir('fpga')
        # make build directory if it doesn't exist
        if not os.path.exists('build'):
            os.makedirs('build')
        # change directory to the build directory
        os.chdir('build')
        # run cmake
        cmake_command = ['cmake', '..']
        with open(os.devnull, 'w') as devnull:
            subprocess.run(cmake_command, check=True, stdout=devnull)
        # run make
        make_command = ['make']
        with open(os.devnull, 'w') as devnull:
            subprocess.run(make_command, check=True, stdout=devnull)
        # change directory back to the parent directory
        os.chdir('../..')

    else:
        flags = []
        if version != 'NCCL':
            flags += ['USE_NCCL=0']
        if version == 'OPENMP' or version == 'SERIAL':
            flags += ['USE_CUDA=0']
        else:
            flags += ['USE_CUDA=1']

        # make clean
        clean_command = ['make', 'clean']
        with open(os.devnull, 'w') as devnull:
            subprocess.run(clean_command, check=True, stdout=devnull)

        # make with flags
        make_command = ['make', '-j4'] + flags
        # print the command being run

        with open(os.devnull, 'w') as devnull:
            subprocess.run(make_command, check=True, stdout=devnull)

def run(version, matrix_file, vector_file, output_file, tolerance, max_iterations):

    if version in tommy_implementations:
        # change directory to the FPGA directory
        os.chdir('fpga')
        # change directory to the build directory
        os.chdir('build')
        # run the executable

        if version ==  'FPGA':
            run_command = ['./single_fpga', matrix_file, vector_file, output_file, max_iterations, tolerance]
        elif version == 'MFPGA':
            run_command = ['./multi_fpga', matrix_file, vector_file, output_file, max_iterations, tolerance, "2"]
        elif version == 'MNFPGA':
            run_command = ['srun', './multi_node', matrix_file, vector_file, output_file, max_iterations, tolerance, str(threads_number)]
        elif version == 'OPENMP':
            run_command = ['./main', max_iterations, tolerance, "0", "1", str(threads_number), matrix_file, vector_file, output_file]
        elif version == 'MPI':
            run_command = ['srun', './mpi', matrix_file, vector_file, output_file, max_iterations, tolerance, str(threads_number)]
        print(' '.join(run_command))
        subprocess.run(run_command, check=True)
        # change directory back to the parent directory
        os.chdir('../..')
    else:
        # make run with arguments
        run_command = ['./bin/conj', matrix_file, vector_file, output_file, max_iterations, tolerance, str(implementation_numbers[version]), f'time_{version}.txt']
        if version == 'NCCL':
            run_command = ['srun'] + run_command
        
        subprocess.run(run_command, check=True)

def check_results(output_file, reference_file, implementation):
    if reference_file is None:
        print('No reference file provided')
        return

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
            if implementation != 'ROWS':
                print(output_matrix[i], "vs",  reference_matrix[i])
    if is_correct:
        print('The output solution is correct')
    else:
        print('The output solution is incorrect')

def calculate_speedup(time_file, reference_time):
    # read the time file
    if not os.path.exists(time_file):
        print('No time file found')
        return -1

    with open(time_file, 'r') as file:
        time = int(file.read())
    time = time if time != 0 else 1
    speedup = reference_time / time
    print('Speedup: ', speedup)
    return time

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Compile and run specific CG solver')

    # Add the version argument "cuBLAS", "MGPU", "Luca GPU", "Tommy GPU", "CPU (OpenMP)", "CPU (Serial)"
    # parser.add_argument('matrix_file', help='The file containing the matrix to solve')
    # parser.add_argument('vector_file', help='The file containing the vector to solve')
    # parser.add_argument('output_file', help='The file to write the solution to')
    parser.add_argument('data_folder', help='The folder containing the data files')
    parser.add_argument('tolerance', help='The tolerance for the solution')
    parser.add_argument('max_iterations', help='The maximum number of iterations to perform')
    parser.add_argument('reference_file', nargs='?', default=None, help='The reference file to compare the output to')
    parser.add_argument('implementation', choices=['NCCL', 'MGPU', 'CUBLAS', 'TILED', 'ROWS', 'OPENMP', 'MPI', 'SERIAL', 'FPGA', 'MFPGA', 'MNFPGA', 'ALL', 'ALL_CPU', 'ALL_FPGA'], help='The version of the program to run')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    df = pd.DataFrame(columns=['implementation', 'matrix_size', 'time'])

    for size in matrix_size:
        matrix = f'{args.data_folder}/matrix{size}.bin'
        rhs = f'{args.data_folder}/rhs{size}.bin'
        output = f'output/output_{size}.bin'
        # make them absolute paths
        matrix = os.path.abspath(matrix)
        rhs = os.path.abspath(rhs)
        output = os.path.abspath(output)
        # if one of them does not exist, skip
        if not os.path.exists(matrix) or not os.path.exists(rhs):
            continue
        if 'ALL' in args.implementation:
            if 'GPU' in args.implementation:
                implementations = ['CUBLAS', 'TILED', 'ROWS', 'MGPU', 'NCCL']
            elif 'CPU' in args.implementation:
                implementations = ['SERIAL', 'OPENMP', 'MPI']
            elif 'FPGA' in args.implementation:
                implementations = ['FPGA', 'MFPGA', 'MNFPGA']
            else:
                implementations = implementation_numbers.keys()
            for implementation in implementations:
                compile(implementation)
                run(implementation, matrix, rhs, output, args.tolerance, args.max_iterations)
                check_results(output, args.reference_file, implementation)
                time = calculate_speedup(f'output/time_{implementation}.txt', reference_time)
                if df.empty:
                    df = pd.DataFrame([[implementation, size, time]], columns=['implementation', 'matrix_size', 'time'])
                else :
                    df = pd.concat([df, pd.DataFrame([[implementation, size, time]], columns=['implementation', 'matrix_size', 'time'])])
        else:
            compile(args.implementation)
            run(args.implementation, matrix, rhs, output, args.tolerance, args.max_iterations)
            check_results(output, args.reference_file, args.implementation)
            time = calculate_speedup(f'output/time_{args.implementation}.txt', reference_time)
            if df.empty:
                df = pd.DataFrame([[args.implementation, size, time]], columns=['implementation', 'matrix_size', 'time'])
            else:
                df = pd.concat([df, pd.DataFrame([[args.implementation, size, time]], columns=['implementation', 'matrix_size', 'time'])])
    
    df.to_csv(f'output/times_{args.implementation}.csv', index=False)
