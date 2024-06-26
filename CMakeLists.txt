#------------------------------------------------------------------------------#
# Copyright 2018 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(Legion_ImageCompositor)

set(BUILD_SHARED_LIBS ON   CACHE BOOL   "Whether or not to build shared libraries instead of static")

option(BUILD_WITH_CUDA "Enable CUDA-based rendering" ON)

#------------------------------------------------------------------------------#
# Install into the CMake pacakge registry if explicitly selected to do so
#------------------------------------------------------------------------------#

set(Legion_DIR $ENV{LG_RT_DIR}/../language/build)
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

# Thrust is required for complex reductions; let cmake find it.
# By including Thrust before the CUDA headers we make sure the Thrust installation
# we pass to cmake will take priority against whatever version is bundled with CUDA.
find_package(Thrust CONFIG)
# If we can't find a standalone Thrust then don't do anything and just let the build
# pick up the version that comes bundled with CUDA.
if(Thrust_FOUND)
  thrust_create_target(Thrust)
endif()

set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON CACHE INTERNAL "Disable the export(PACKAGE) command.")
option(Legion_USE_PACKAGE_REGISTRY "Register the build tree with the CMake package registry" OFF)
mark_as_advanced(Legion_USE_PACKAGE_REGISTRY)
if(Legion_USE_PACKAGE_REGISTRY)
  set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY OFF)
endif()
export(PACKAGE Legion)

message("LEGION_DIR is $ENV{LEGION_DIR}")

include_directories( $ENV{LEGION_DIR}/bindings/regent )
include_directories( $ENV{LEGION_DIR}/language/build/runtime )
include_directories( $ENV{LEGION_DIR}/runtime/legion )
include_directories( $ENV{LEGION_DIR}/runtime/realm )
include_directories( include )

include (FindPkgConfig)
if (NOT PKG_CONFIG_FOUND)
    message (FATAL_ERROR "pkg-config not found")
endif ()

# check for libpng
pkg_check_modules (LIBPNG libpng16 REQUIRED)
if (NOT LIBPNG_FOUND)
    message(FATAL_ERROR "You don't seem to have libpng16 development libraries installed")
else ()
    include_directories (${LIBPNG_INCLUDE_DIRS})
    link_directories (${LIBPNG_LIBRARY_DIRS})
    link_libraries (${LIBPNG_LIBRARIES})
endif ()
# add_executable (app_png ${_MYSOURCES} ${LIBPNG_LINK_FLAGS})

# include_directories( deps/libpng/include )

add_subdirectory(src)


#------------------------------------------------------------------------------#
# Compile OVR Visualization Library
#------------------------------------------------------------------------------#

# include rendering library
set(OVR_BUILD_OPENGL OFF CACHE BOOL "" FORCE)
set(OVR_BUILD_SCENE_USD OFF CACHE BOOL "" FORCE)

find_package(ospray 2.0 REQUIRED)
find_package(TBB REQUIRED)

if(BUILD_WITH_CUDA)
    set(OVR_BUILD_CUDA ON CACHE BOOL "" FORCE)
else()
    set(OVR_BUILD_CUDA OFF CACHE BOOL "" FORCE)
endif()
set(OVR_BUILD_DEVICE_OPTIX7 OFF CACHE BOOL "" FORCE)
set(OVR_BUILD_DEVICE_OSPRAY ON  CACHE BOOL "" FORCE)
add_subdirectory(renderer)
