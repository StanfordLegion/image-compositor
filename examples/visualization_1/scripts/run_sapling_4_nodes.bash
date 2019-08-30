#!bin/bash
mpirun -n 4 -npernode 1 -H n0000,n0001,n0002,n0003 --bind-to none ./visualization_1 -ll:cpu 4 -ll:csize 2048

