#!/bin/bash -l
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=1                         # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=00:03:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=p200301                  # project account
#SBATCH --cpus-per-task=256                # CORES per task

module load ifpgasdk && module load 520nmx && module load CMake && module load intel && module load deploy/EasyBuild
cd build
git pull
make main
./main 30000 1e-16 1 0 50 ignore ignore output.bin
./main 50000 1e-16 1 0 100 ignore ignore output.bin
./main 10000 1e-16 0 1 50 ignore ignore output.bin
./main 10000 1e-16 0 1 100 ignore ignore output.bin
./main 10000 1e-16 0 1 200 ignore ignore output.bin
./main 20000 1e-16 0 1 50 ignore ignore output.bin
./main 20000 1e-16 0 1 100 ignore ignore output.bin
./main 20000 1e-16 0 1 200 ignore ignore output.bin
./main 30000 1e-16 0 1 50 ignore ignore output.bin
./main 30000 1e-16 0 1 100 ignore ignore output.bin
./main 30000 1e-16 0 1 200 ignore ignore output.bin
./main 50000 1e-16 0 1 50 ignore ignore output.bin
./main 50000 1e-16 0 1 100 ignore ignore output.bin
./main 50000 1e-16 0 1 200 ignore ignore output.bin
./main 60000 1e-16 0 1 50 ignore ignore output.bin
./main 60000 1e-16 0 1 100 ignore ignore output.bin
./main 60000 1e-16 0 1 200 ignore ignore output.bin
./main 70000 1e-16 0 1 50 ignore ignore output.bin
./main 70000 1e-16 0 1 100 ignore ignore output.bin
./main 70000 1e-16 0 1 200 ignore ignore output.bin
./main 80000 1e-16 0 1 50 ignore ignore output.bin
./main 80000 1e-16 0 1 100 ignore ignore output.bin
./main 80000 1e-16 0 1 200 ignore ignore output.bin
./main 90000 1e-16 0 1 50 ignore ignore output.bin
./main 90000 1e-16 0 1 100 ignore ignore output.bin
./main 90000 1e-16 0 1 200 ignore ignore output.bin
./main 100000 1e-16 0 1 50 ignore ignore output.bin
./main 100000 1e-16 0 1 100 ignore ignore output.bin
./main 100000 1e-16 0 1 200 ignore ignore output.bin
./main 150000 1e-16 0 1 50 ignore ignore output.bin
./main 150000 1e-16 0 1 100 ignore ignore output.bin
./main 150000 1e-16 0 1 200 ignore ignore output.bin