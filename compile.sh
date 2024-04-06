#!/bin/bash

GPU_ARCH=$1
if [[ -z "$GPU_ARCH" ]]; then
   echo "No CUDA"
   CMAKE_CUDA_ARGS="-DBUILD_WITH_CUDA=OFF "
else
   CMAKE_CUDA_ARGS="-DGDT_CUDA_ARCHITECTURES=${GPU_ARCH} -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCH} -DBUILD_WITH_CUDA=ON "
fi

export LEGION_DIR=$(pwd)/../legion
export LG_RT_DIR=$(pwd)/../legion/runtime

mkdir -p deps
cd deps

DEPS=$(pwd)

if [ ! -d "$DEPS/libpng" ]; then
   wget https://download.sourceforge.net/libpng/libpng-1.6.39.tar.gz
   tar xvzf libpng-1.6.39.tar.gz
   cd libpng-1.6.39
   ./configure --prefix=$DEPS/libpng
   make -j
   make PREFIX=$DEPS/libpng install
   cd ..
fi

if [ ! -d "ospray-2.9.0.x86_64.linux" ]; then
   wget https://github.com/ospray/ospray/releases/download/v2.9.0/ospray-2.9.0.x86_64.linux.tar.gz
   tar xzvf ospray-2.9.0.x86_64.linux.tar.gz
fi

if [ ! -d "oneapi-tbb-2021.4.0" ]; then
   wget https://github.com/oneapi-src/oneTBB/releases/download/v2021.4.0/oneapi-tbb-2021.4.0-lin.tgz
   tar xzvf oneapi-tbb-2021.4.0-lin.tgz
fi

# if [ ! -d "open-volume-renderer" ]; then
#    mkdir open-volume-renderer
#    cd open-volume-renderer
#    tar xzvf ../../open-volume-renderer.tar.gz
#    cd ..
# fi

cd ..

mkdir -p build
cd build

pwd

# note: BUILD_WITH_CUDA means whether Legion has CUDA with it or not
#       OVR_BUILD_DEVICE_OPTIX7 means whether we want to enable the OptiX renderer
PKG_CONFIG_PATH=$DEPS/libpng/lib/pkgconfig:$PKG_CONFIG_PATH \
VERBOSE=0 cmake .. ${CMAKE_CUDA_ARGS} \
   -Dospray_DIR=${DEPS}/ospray-2.9.0.x86_64.linux/lib/cmake/ospray-2.9.0/ \
   -DTBB_DIR=${DEPS}/oneapi-tbb-2021.4.0/lib/cmake/tbb

VERBOSE=0 cmake --build . --config Debug --parallel 16
