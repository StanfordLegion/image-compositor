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

cmake_minimum_required(VERSION 3.1)
project(Legion)

if(BUILD_WITH_CUDA)
  enable_language(CUDA)
endif()

set(Legion_DIR $ENV{LG_RT_DIR}/../language/build)
if(NOT Legion_SOURCE_DIR)
  find_package(Legion REQUIRED)
endif()

include_directories(../include $ENV{LEGION_DIR}/runtime $ENV{LEGION_DIR}/runtime/legion $ENV{LEGION_DIR}/runtime/realm)

list(APPEND FRAMEWORK_SRC
  image.cc
  image_reduction.cc
  image_reduction_composite.cc
  image_reduction_mapper.cc
)
if(BUILD_WITH_CUDA)
  set_source_files_properties(${FRAMEWORK_SRC} PROPERTIES LANGUAGE CUDA)
endif()



add_library(ImageCompositor SHARED ${FRAMEWORK_SRC})
# target_link_libraries(ImageCompositor PUBLIC ../deps/libpng/lib/libpng16.so)
