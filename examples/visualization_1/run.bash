#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF_ROOT/lib
mpirun -n 1 -npernode 4 ./visualization_1 -ll:cpu 4 -ll:csize 6144
