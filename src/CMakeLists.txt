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

include(cuda_helpers)

include_directories(../include $ENV{LEGION_DIR}/runtime $ENV{LEGION_DIR}/runtime/legion $ENV{LEGION_DIR}/runtime/realm)

list(APPEND FRAMEWORK_SRC
  image.cc
  image_reduction.cc
  image_reduction_composite.cc
  image_reduction_mapper.cc
)

add_library(ImageCompositor SHARED ${FRAMEWORK_SRC})

target_link_libraries(ImageCompositor PRIVATE Legion::Legion)
# target_link_libraries(ImageCompositor PRIVATE ../deps/libpng/lib/libpng16.so)
target_compile_options(ImageCompositor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_BUILD_WARNING_FLAGS}>)

if(Legion_USE_CUDA)
  set_source_files_properties(${FRAMEWORK_SRC} PROPERTIES LANGUAGE CUDA)
  set_target_cuda_standard(ImageCompositor STANDARD ${Legion_CUDA_STANDARD})
  set_target_cuda_architectures(ImageCompositor ARCHITECTURES ${Legion_CUDA_ARCH})
endif()
# set_target_cuda_warnings_and_errors(ImageCompositor WARN_AS_ERROR ${Legion_BUILD_WARN_AS_ERROR})
