#!/bin/bash

export LEGION_DIR=$(pwd)/../../../legion
DEPS=$(pwd)/../../deps

mkdir -p build
cd build

# note: BUILD_WITH_CUDA means whether Legion has CUDA with it or not
#       OVR_BUILD_DEVICE_OPTIX7 means whether we want to enable the OptiX renderer
PKG_CONFIG_PATH=$DEPS/libpng/lib/pkgconfig:$PKG_CONFIG_PATH VERBOSE=0 cmake ../ \
    -DBUILD_WITH_CUDA=ON -DOVR_BUILD_DEVICE_OPTIX7=OFF \
    -Dospray_DIR=${DEPS}/ospray-2.9.0.x86_64.linux/lib/cmake/ospray-2.9.0/ \
    -DTBB_DIR=${DEPS}/oneapi-tbb-2021.4.0/lib/cmake/tbb \

VERBOSE=0 cmake --build . --config Debug --parallel 16
