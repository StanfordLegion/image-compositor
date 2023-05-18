cmake_minimum_required(VERSION 3.1)
project(visualization_3 LANGUAGES C CXX)

option(BUILD_WITH_CUDA "Enable OptiX rendering" ON)


# adapted from https://stackoverflow.com/a/69353718
if(NOT CMAKE_CUDA_ARCHITECTURES)
  find_package(CUDA REQUIRED)
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()
# print debug messages
message(STATUS "Automatically detected GPU architectures: ${CUDA_ARCH_LIST}")
# we can only enable CUDA if CMAKE_CUDA_ARCHITECTURES is set
if (CMAKE_CUDA_ARCHITECTURES)
  enable_language(CUDA)
else()
  message(FATAL_ERROR "Cannot utomatically detected GPU architecture")
endif()


# the `pkg_check_modules` function is created with this call
find_package(PkgConfig REQUIRED)
pkg_check_modules(OSMesa REQUIRED IMPORTED_TARGET osmesa)
pkg_check_modules(GLU REQUIRED IMPORTED_TARGET glu)
find_package(MPI REQUIRED)
find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fpermissive")

# now find legion
set(Legion_DIR $ENV{LG_RT_DIR}/../language/build)
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

# build regent
add_custom_target(run ALL DEPENDS libviz_3.so)
add_custom_command(OUTPUT libviz_3.so
  COMMAND regent.py ${CMAKE_CURRENT_LIST_DIR}/visualization_3.rg
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/visualization_3.rg)

add_library(libviz_3 SHARED IMPORTED)
set_target_properties(libviz_3 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libviz_3.so)

if(NOT IMAGE_COMPOSITOR_DIR)
  # message(FATAL_ERROR "Path IMAGE_COMPOSITOR_DIR not set.")
  message(STATUS "Using default IMAGE_COMPOSITOR_DIR")
  set(IMAGE_COMPOSITOR_DIR ../../../build)
endif()


include_directories(${CMAKE_CURRENT_LIST_DIR}/../../include)
# add_library(imageCompositor STATIC IMPORTED)
# SET_TARGET_PROPERTIES(imageCompositor PROPERTIES IMPORTED_LOCATION ${IMAGE_COMPOSITOR_DIR}/src/libImageCompositor.a)
add_library(imageCompositor SHARED IMPORTED)
SET_TARGET_PROPERTIES(imageCompositor PROPERTIES IMPORTED_LOCATION ${IMAGE_COMPOSITOR_DIR}/src/libImageCompositor.so)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../deps/open-volume-renderer/common/gdt/cmake")
find_package(TBB    REQUIRED)
find_package(ospray REQUIRED)
if(OVR_BUILD_DEVICE_OPTIX7)
  find_package(OptiX  REQUIRED)
  find_package(CUDA   REQUIRED)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR}/../../deps/open-volume-renderer)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../deps/open-volume-renderer/extern)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../deps/open-volume-renderer/extern/gdt)
add_library(renderlib SHARED IMPORTED)
set_target_properties(renderlib PROPERTIES IMPORTED_LOCATION ${IMAGE_COMPOSITOR_DIR}/deps/open-volume-renderer/librenderlib.so)
target_link_libraries(renderlib INTERFACE TBB::tbb ospray::ospray stdc++fs)
target_compile_definitions(renderlib INTERFACE OVR_BUILD_OSPRAY=1)
if(OVR_BUILD_DEVICE_OPTIX7)
  target_link_libraries(renderlib INTERFACE OptiX::optix7)
  target_compile_definitions(renderlib INTERFACE 
    OVR_ENABLE_CUDA_DEVICES=1
    OVR_BUILD_OPTIX7=1
  )
endif()

add_executable(visualization_3 main.cc render.cc render.h image.cc image.h visualization_3_mapper.cc)
add_dependencies(visualization_3 run)

find_package(ZLIB REQUIRED)


if(BUILD_WITH_CUDA)
  set_source_files_properties(render.cc PROPERTIES LANGUAGE CUDA)
endif()

target_include_directories(visualization_3 PRIVATE ${MPI_C_INCLUDE_PATH})
target_link_libraries(visualization_3
PUBLIC
  PkgConfig::GLU PkgConfig::OSMesa PNG::PNG
  Legion::LegionRuntime Legion::Legion 
  libviz_3 imageCompositor
  ${MPI_CXX_LIBRARIES}
  z
PRIVATE
  renderlib
)
