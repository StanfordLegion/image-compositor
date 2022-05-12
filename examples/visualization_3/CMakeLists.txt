cmake_minimum_required(VERSION 3.1)
project(visualization_3 LANGUAGES C CXX )

set(Legion_DIR $ENV{LG_RT_DIR}/../language/build)

if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

# the `pkg_check_modules` function is created with this call
find_package(PkgConfig REQUIRED)

pkg_check_modules(OSMesa REQUIRED IMPORTED_TARGET osmesa)
pkg_check_modules(GLU REQUIRED IMPORTED_TARGET glu)

#find_package(ParaView REQUIRED)
#find_package(GLU REQUIRED)
#find_package(OSMesa REQUIRED)
find_package(MPI REQUIRED)
find_package(PNG REQUIRED)
#find_package(VTK COMPONENTS VTK::IOXML)
find_package(VTK COMPONENTS IOXML)
find_package(ospray 2.0 REQUIRED)

add_subdirectory(ovr EXCLUDE_FROM_ALL)
# include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
# include_directories(${OptiX_INCLUDE})

set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fpermissive")

add_custom_target(run ALL
  DEPENDS libviz_3.so)

add_custom_command(OUTPUT libviz_3.so
  COMMAND regent.py ${CMAKE_CURRENT_LIST_DIR}/visualization_3.rg
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/visualization_3.rg)

add_library(libviz_3 SHARED IMPORTED)
set_target_properties(libviz_3 PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libviz_3.so)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../../include)
# add_library(imageCompositor STATIC IMPORTED)
# SET_TARGET_PROPERTIES(imageCompositor PROPERTIES IMPORTED_LOCATION ../../build/src/libImageCompositor.a)
add_library(imageCompositor SHARED IMPORTED)
SET_TARGET_PROPERTIES(imageCompositor PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/../../build/src/libImageCompositor_.so)

add_executable(visualization_3 main.cc render.cc render.h image.cc image.h visualization_3_mapper.cc)
add_dependencies(visualization_3 run)
target_include_directories(visualization_3 PRIVATE ${MPI_C_INCLUDE_PATH})
  
target_link_libraries(visualization_3
PUBLIC
  PkgConfig::GLU PkgConfig::OSMesa Legion::LegionRuntime Legion::Legion libviz_3 imageCompositor
  ${MPI_CXX_LIBRARIES}
  PNG::PNG

INTERFACE
  VTK::PythonUsed

PRIVATE
  renderlib
  #ParaView::PythonCatalyst
  VTK::CommonDataModel
  ${VTK_LIBRARIES}
  VTK::IOXML
  )

# vtk_module_autoinit is needed
vtk_module_autoinit(
  TARGETS visualization_3e
  MODULES ${VTK_LIBRARIES}
)
