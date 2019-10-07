#!/bin/bash
export LEGION_FREEZE_ON_ERROR=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LEGION_DIR/bindings/regent/:$HDF_ROOT/lib
mpirun -n 1 -npernode 8 ../visualization_2.exec -ll:cpu 4 -ll:csize 6144

