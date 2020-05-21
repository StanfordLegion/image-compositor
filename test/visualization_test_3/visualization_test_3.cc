/* Copyright 2017 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "visualization_reductions.h"

const int numDomainNodesX = 2;
const int numDomainNodesY = 2;
const int numDomainNodesZ = 1;
const int numDomainNodes = numDomainNodesX * numDomainNodesY * numDomainNodesZ;


void top_level_task(const Legion::Task *task,
                    const std::vector<Legion::PhysicalRegion> &regions,
                    Legion::Context ctx, Legion::Runtime *runtime) {


  {
    // test with large images
    Legion::Visualization::ImageDescriptor imageDescriptor = { 1920, 1080, numDomainNodes };
    Legion::Visualization::ImageReduction imageReduction(imageDescriptor, ctx, runtime, 0);
    Legion::Visualization::testAssociative(imageReduction, imageDescriptor, ctx, runtime, Legion::Visualization::depthFuncs[0], 0, 0, Legion::Visualization::blendEquations[0]);
    Legion::Visualization::testNonassociative(imageReduction, imageDescriptor, ctx, runtime, Legion::Visualization::depthFuncs[0], 0, 0, Legion::Visualization::blendEquations[0]);
  }

}




int main(int argc, char *argv[]) {

  Legion::Visualization::ImageReduction::preinitializeBeforeRuntimeStarts();  
  Legion::HighLevelRuntime::set_top_level_task_id(Legion::Visualization::TOP_LEVEL_TASK_ID);

  {
    Legion::TaskVariantRegistrar registrar(Legion::Visualization::TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    Legion::TaskVariantRegistrar registrar(Legion::Visualization::GENERATE_IMAGE_DATA_TASK_ID, "generate_image_data_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<Legion::Visualization::generate_image_data_task>(registrar, "generate_image_data_task");
  }

  {
    Legion::TaskVariantRegistrar registrar(Legion::Visualization::VERIFY_COMPOSITED_IMAGE_DATA_TASK_ID, "verify_composited_image_data_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    registrar.set_leaf();
    Legion::Runtime::preregister_task_variant<int, Legion::Visualization::verify_composited_image_data_task>(registrar, "verify_composited_image_data_task");
  }

  return Legion::HighLevelRuntime::start(argc, argv);
}
