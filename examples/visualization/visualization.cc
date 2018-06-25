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


#include <iostream>

#include "legion.h"
#include "legion_visualization.h"

#include "usec_timer.h"


using namespace Legion;
using namespace Legion::Visualization;


enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  RENDER_TASK_ID,
};



static void simulateTimeStep(int t) {
  // tbd
}



static void paintRegion(ImageSize imageSize,
                        ImageReduction::PixelField *r,
                        ImageReduction::PixelField *g,
                        ImageReduction::PixelField *b,
                        ImageReduction::PixelField *a,
                        ImageReduction::PixelField *z,
                        ImageReduction::PixelField *userdata,
                        ImageReduction::Stride stride,
                        int layer) {
  
  ImageReduction::PixelField zValue = layer;
  for(int row = 0; row < imageSize.height; ++row) {
    for(int column = 0; column < imageSize.width; ++column) {
      *r = layer;
      *g = layer;
      *b = layer;
      *a = layer;
      *z = layer;
      *userdata = layer;
      r += stride[ImageReduction::FID_FIELD_R][0];
      g += stride[ImageReduction::FID_FIELD_G][0];
      b += stride[ImageReduction::FID_FIELD_B][0];
      a += stride[ImageReduction::FID_FIELD_A][0];
      z += stride[ImageReduction::FID_FIELD_Z][0];
      userdata += stride[ImageReduction::FID_FIELD_USERDATA][0];
      zValue = (zValue + 1);
      zValue = (zValue >= imageSize.numImageLayers) ? 0 : zValue;
    }
  }
}

void render_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, HighLevelRuntime *runtime) {
  
  UsecTimer render(Legion::Visualization::ImageReduction::describe_task(task) + ":");
  render.start();
  PhysicalRegion image = regions[0];
  ImageSize imageSize = ((ImageSize *)task->args)[0];
  
  ImageReduction::PixelField *r, *g, *b, *a, *z, *userdata;
  ImageReduction::Stride stride;
  int layer = task->get_unique_id() % imageSize.numImageLayers;
  ImageReduction::create_image_field_pointers(imageSize, image, r, g, b, a, z, userdata, stride, runtime, ctx);
  paintRegion(imageSize, r, g, b, a, z, userdata, stride, layer);
  render.stop();
  cout << render.to_string() << endl;
}



void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {
  

#ifdef IMAGE_SIZE
  ImageSize imageSize = (ImageSize){ IMAGE_SIZE };
  
#else
  const int width = 3840;
  const int height = 2160;
  const int numSimulationTasks = 4;
  const int numFragmentsPerLayer = 8;
  
  ImageSize imageSize = (ImageSize){ width, height, numSimulationTasks, numFragmentsPerLayer };
#endif
  
  std::cout << "ImageSize (" << imageSize.width << "," << imageSize.height
  << ") x " << imageSize.numImageLayers << " layers " << imageSize.numFragmentsPerLayer << " frags/layer" << std::endl;
  
  ImageReduction imageReduction(imageSize, ctx, runtime);
  imageReduction.set_blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  imageReduction.set_blend_equation(GL_FUNC_ADD);
  
  {
    UsecTimer overall("overall time:");
    overall.start();
    UsecTimer frame("frame time:");
    UsecTimer reduce("reduce time:");
    Future displayFuture;
    
    const int numTimeSteps = 5;
    
    for(int t = 0; t < numTimeSteps; ++t) {
      
      frame.start();
      simulateTimeStep(t);
      FutureMap renderFutures = imageReduction.launch_index_task_by_depth(RENDER_TASK_ID, runtime, ctx);
      renderFutures.wait_all_results();
      
      reduce.start();
      FutureMap reduceFutures = imageReduction.reduce_associative_commutative();
      reduceFutures.wait_all_results();
      reduce.stop();
      
      displayFuture = imageReduction.display(t);
      displayFuture.wait();
      frame.stop();
    }
    
    overall.stop();
    
    std::cout << reduce.to_string() << std::endl;
    std::cout << frame.to_string() << std::endl;
    std::cout << overall.to_string() << std::endl;
    
  }
}





int main(const int argc, char *argv[]) {
  
  Legion::Visualization::ImageReduction::initialize();
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
                                                         Processor::LOC_PROC, true/*single*/, false/*index*/,
                                                         AUTO_GENERATE_ID, TaskConfigOptions(false/*leaf*/), "topLevelTask");
  HighLevelRuntime::register_legion_task<render_task>(RENDER_TASK_ID,
                                                      Processor::LOC_PROC, false/*single*/, true/*index*/,
                                                      AUTO_GENERATE_ID, TaskConfigOptions(true/*leaf*/), "renderTask");
  
  return HighLevelRuntime::start(argc, argv);
}
