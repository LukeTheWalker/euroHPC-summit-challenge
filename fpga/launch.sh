#!/bin/bash -l
#SBATCH --qos=default                      # SLURM qos
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=1                         # number of tasks
#SBATCH --ntasks-per-node=1                # number of tasks per node
#SBATCH --time=10:00:00                    # time (HH:MM:SS)
#SBATCH --partition=cpu                    # partition
#SBATCH --account=p200301                  # project account
#SBATCH --cpus-per-task=8                  # CORES per task

module load ifpgasdk
module load 520nmx
cd src/fpga
aoc -parallel=8 -v -report -board=p520_hpc_m210h_g3x16 -fp-relaxed -DINTEL_CL -o CG_improved_v2 kernels.cl

