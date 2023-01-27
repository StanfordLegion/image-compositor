#!/bin/bash
export REALM_FREEZE_ON_ERROR=1
export REALM_USE_CUDART_HIJACK=0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/qwu/image-compositor/deps/ospray-2.9.0.x86_64.linux/lib:/home/qwu/image-compositor/deps/oneapi-tbb-2021.4.0/lib/intel64/gcc4.8

echo $OMPI_COMM_WORLD_LOCAL_RANK
cd /home/qwu/image-compositor/examples/visualization_3/build-cuda
CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK ./visualization_3 -pipeline volume.py \
    -logfile run_%.logls -level mapper=2,inst=1 -ll:gpu 1
