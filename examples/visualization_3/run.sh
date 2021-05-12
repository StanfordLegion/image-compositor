export REALM_FREEZE_ON_ERROR=1
mpirun -np 2 visualization_3 -pipeline ../volume.py -logfile run_%.log
