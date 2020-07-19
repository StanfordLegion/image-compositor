cmake_minimum_required(VERSION 3.1)
project(visualization_3)

set(Legion_DIR $ENV{LG_RT_DIR}/../language/build)

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

find_package(ParaView REQUIRED)
find_package(MPI REQUIRED)
find_package(PNG REQUIRED)

set(CMAKE_CXX_FLAGS "-fpermissive")

add_custom_target(run ALL
  DEPENDS libviz_3.so)

add_custom_command(OUTPUT libviz_3.so
  COMMAND regent.py visualization_3.rg
  DEPENDS visualization_3.rg)

add_library(libviz_3 SHARED IMPORTED)
set_target_properties(libviz_3 PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libviz_3.so)

include_directories(../../include)
# add_library(imageCompositor STATIC IMPORTED)
# SET_TARGET_PROPERTIES(imageCompositor PROPERTIES IMPORTED_LOCATION ../../build/src/libImageCompositor.a)
add_library(imageCompositor SHARED IMPORTED)
SET_TARGET_PROPERTIES(imageCompositor PROPERTIES IMPORTED_LOCATION ../../build/src/libImageCompositor_.so)

add_executable(visualization_3 main.cc render.cc render.h image.cc image.h visualization_3_mapper.cc)
add_dependencies(visualization_3 run)
target_include_directories(visualization_3 PRIVATE ${MPI_C_INCLUDE_PATH})

target_link_libraries(visualization_3
  PUBLIC GLU OSMesa Legion::LegionRuntime Legion::Legion libviz_3 imageCompositor ${MPI_CXX_LIBRARIES} PNG::PNG
  INTERFACE VTK::PythonUsed PRIVATE ParaView::PythonCatalyst VTK::CommonDataModel)
