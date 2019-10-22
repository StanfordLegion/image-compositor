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



static void paintRegion(ImageDescriptor imageDescriptor,
                        ImageReduction::PixelField *r,
                        ImageReduction::PixelField *g,
                        ImageReduction::PixelField *b,
                        ImageReduction::PixelField *a,
                        ImageReduction::PixelField *z,
                        ImageReduction::PixelField *userdata,
                        ImageReduction::Stride stride,
                        int layer) {

  ImageReduction::PixelField zValue = layer;
  for(int row = 0; row < imageDescriptor.height; ++row) {
    for(int column = 0; column < imageDescriptor.width; ++column) {
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
      zValue = (zValue >= imageDescriptor.numImageLayers) ? 0 : zValue;
    }
  }
}

void render_task(const Task *task,
                 const std::vector<PhysicalRegion> &regions,
                 Context ctx, HighLevelRuntime *runtime) {

  Processor processor = runtime->get_executing_processor(ctx);
  Machine::ProcessorQuery query(Machine::get_machine());
  query.only_kind(processor.kind());
  if(processor.id == query.first().id) {

    UsecTimer render(Legion::Visualization::ImageReduction::describe_task(task) + ":");
    render.start();
    PhysicalRegion image = regions[0];
    ImageDescriptor imageDescriptor = ((ImageDescriptor *)task->args)[0];

    ImageReduction::PixelField *r, *g, *b, *a, *z, *userdata;
    ImageReduction::Stride stride;
    int layer = task->get_unique_id() % imageDescriptor.numImageLayers;
    ImageReduction::create_image_field_pointers(imageDescriptor, image, r, g, b, a, z, userdata, stride, runtime, ctx, true);
    paintRegion(imageDescriptor, r, g, b, a, z, userdata, stride, layer);
    render.stop();
    cout << render.to_string() << endl;
  }
}



void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime) {


#ifdef IMAGE_SIZE
  ImageDescriptor imageDescriptor = (ImageDescriptor){ IMAGE_SIZE };

#else
  const int width = 1920;
  const int height = 1080;
  const int numSimulationTasks = 4;

  ImageDescriptor imageDescriptor = (ImageDescriptor){ width, height, numSimulationTasks };
#endif

  std::cout << "ImageDescriptor (" << imageDescriptor.width << "," << imageDescriptor.height
  << ") x " << imageDescriptor.numImageLayers << " layers " << std::endl;

  ImageReduction imageReduction(imageDescriptor, ctx, runtime);
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
      FutureMap renderFutures = imageReduction.launch_task_composite_domain(RENDER_TASK_ID, runtime, ctx);
      renderFutures.wait_all_results();

      reduce.start();
      float cameraDirection[3] = { 0, 0, 1 };
      FutureMap reduceFutures = imageReduction.reduceImages(ctx, cameraDirection);
      reduceFutures.wait_all_results();
      reduce.stop();

      displayFuture = imageReduction.display(t, ctx);
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

  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    Legion::TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    Legion::TaskVariantRegistrar registrar(RENDER_TASK_ID, "render_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<top_level_task>(registrar, "render_task");
  }

  return HighLevelRuntime::start(argc, argv);
}
