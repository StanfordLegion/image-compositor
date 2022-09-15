export REALM_FREEZE_ON_ERROR=1
export REALM_USE_CUDART_HIJACK=0
# mpirun -np 2 visualization_3 -pipeline ../volume.py -logfile run_%.log
# make -j && mpirun -np 2 visualization_3 \
#     -pipeline volume.py -logfile run_%.logls -level mapper=2,inst=1 \

cd /home/qwu/image-compositor/examples/visualization_3/build-cuda
CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK ./visualization_3 -pipeline volume.py \
    -logfile run_%.logls -level mapper=2,inst=1
