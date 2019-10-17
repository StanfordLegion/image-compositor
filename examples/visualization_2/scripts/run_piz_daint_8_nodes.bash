#!/bin/bash -l
#SBATCH --job-name=visualization_2
#SBATCH --mail-user=aheirich@stanford.edu
#SBATCH --time=00:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export REALM_BACKTRACE=1
#export LEGION_BACKTRACE=1
export GASNET_BACKTRACE=1
#export REALM_FREEZE_ON_ERROR=1
#export LEGION_FREEZE_ON_ERROR=1

source ~/image-compositor/setup.bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LEGION_DIR/bindings/regent/

srun -C gpu -N 8 ~/image-compositor/image-compositor/examples/visualization_2/visualization_2.exec -ll:cpu 16 -ll:csize 8192
