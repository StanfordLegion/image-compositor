#!/bin/bash

mkdir -p deps
cd deps

DEPS=$(pwd)

# wget https://github.com/ospray/ospray/releases/download/v2.9.0/ospray-2.9.0.x86_64.linux.tar.gz
# tar xzvf ospray-2.9.0.x86_64.linux.tar.gz

# wget https://github.com/oneapi-src/oneTBB/releases/download/v2021.4.0/oneapi-tbb-2021.4.0-lin.tgz
# tar xzvf oneapi-tbb-2021.4.0-lin.tgz

mkdir open-volume-renderer
cd open-volume-renderer
unzip ../../open-volume-renderer.zip
cd ..

cd ..

mkdir -p build
cd build

pwd
cmake ../ -Dospray_DIR=${DEPS}/ospray-2.9.0.x86_64.linux/lib/cmake/ospray-2.9.0/ -DTBB_DIR=${DEPS}/oneapi-tbb-2021.4.0/lib/cmake/tbb
cmake --build . --config Debug --parallel 16
