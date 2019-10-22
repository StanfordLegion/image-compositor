#!/bin/bash -l
#SBATCH --job-name=visualization_1
#SBATCH --mail-user=aheirich@stanford.edu
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export REALM_BACKTRACE=1
export GASNET_BACKTRACE=1
#export REALM_FREEZE_ON_ERROR=1

source ~/image-compositor/setup.bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LEGION_DIR/bindings/regent/


srun -C gpu -N 4 $IMAGE_COMPOSITOR_DIR/examples/visualization_1/visualization_1 -ll:cpu 16 -ll:csize 2048
