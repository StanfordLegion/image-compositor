#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF_ROOT/lib
mpirun -n 4 ./visualization_test_2 -ll:cpu 4 -ll:csize 2048 
