export REALM_FREEZE_ON_ERROR=1
export REALM_USE_CUDART_HIJACK=0
# mpirun -np 2 visualization_3 -pipeline ../volume.py -logfile run_%.log
 
make -j && mpirun -np 2 visualization_3 \
    -pipeline volume.py -logfile run_%.logls -level mapper=2,inst=1 \
