#!/bin/bash
mpirun -n 1 -npernode 4 ./visualization_reductions -ll:cpu 4 -ll:csize 2048 
