#!/bin/bash

mkdir -p deps
cd deps

DEPS=$(pwd)

# wget https://github.com/ospray/ospray/releases/download/v2.9.0/ospray-2.9.0.x86_64.linux.tar.gz
# tar xzvf ospray-2.9.0.x86_64.linux.tar.gz

# wget https://github.com/oneapi-src/oneTBB/releases/download/v2021.4.0/oneapi-tbb-2021.4.0-lin.tgz
# tar xzvf oneapi-tbb-2021.4.0-lin.tgz

# mkdir open-volume-renderer
# cd open-volume-renderer
# tar xzvf ../../open-volume-renderer.tar.gz
# cd ..

cd ..

mkdir -p build
cd build

pwd

# note: BUILD_WITH_CUDA means whether Legion has CUDA with it or not
#       OVR_BUILD_DEVICE_OPTIX7 means whether we want to enable the OptiX renderer
cmake ../ \
    -DBUILD_WITH_CUDA=ON -DOVR_BUILD_DEVICE_OPTIX7=OFF \
    -Dospray_DIR=${DEPS}/ospray-2.9.0.x86_64.linux/lib/cmake/ospray-2.9.0/ \
    -DTBB_DIR=${DEPS}/oneapi-tbb-2021.4.0/lib/cmake/tbb

cmake --build . --config Debug --parallel 16
