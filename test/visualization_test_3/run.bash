#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDf_ROOT/lib
mpirun -n 1 -npernode 4 ./visualization_test_3 -ll:cpu 4 -ll:csize 2048 
