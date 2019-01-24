#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LEGION_DIR/language/hdf/install/lib
mpirun -n 1 -npernode 4 ./visualization_test_2 -ll:cpu 4 -ll:csize 2048 
