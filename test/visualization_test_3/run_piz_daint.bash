#!/bin/bash -l
#SBATCH --job-name="visualization_test_3"
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

source ~/setup.bash
source ~/image-compositor/setup.bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF_ROOT/lib

srun ./visualization_test_3 -ll:cpu 4 -ll:csize 4096 
