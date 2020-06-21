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
#include <unistd.h>

#include "legion.h"
#include "legion_visualization.h"
#include "image_reduction_mapper.h"

using namespace Legion;
using namespace Legion::Visualization;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  RENDER_TASK_ID,
};

static void simulateTimeStep(int t) {
  // tbd
}

  typedef const FieldAccessor<READ_WRITE, ImageReduction::PixelField,
    image_region_dimensions,
    coord_t, Realm::AffineAccessor<ImageReduction::PixelField,
    image_region_dimensions, coord_t> > RWAccessor;


static void paintRegion(ImageDescriptor imageDescriptor,
                        RWAccessor r,
                        RWAccessor g,
                        RWAccessor b,
                        RWAccessor a,
                        RWAccessor z,
                        RWAccessor userdata,
                        int layer) {

  std::cout << "paint region with value " << layer << std::endl;
  Legion::Visualization::ImageReduction::PixelField zValue = layer;
  Legion::Visualization::ImageReduction::PixelField value = 1.0 / (1 + layer);
  for(int row = 0; row < imageDescriptor.height; ++row) {
    for(int column = 0; column < imageDescriptor.width; ++column) {
      Realm::Point<image_region_dimensions> p(column, row, layer);
      r[p] = value;
      g[p] = value;
      b[p] = value;
      a[p] = value;
      z[p] = zValue;
      userdata[p] = value;
      zValue = (zValue + 1);
      zValue = (zValue >= imageDescriptor.numImageLayers) ? 0 : zValue;
    }
  }
}

void render_task(const Legion::Task *task,
                 const std::vector<Legion::PhysicalRegion> &regions,
                 Legion::Context ctx, Legion::Runtime *runtime) {

  Legion::Processor processor = runtime->get_executing_processor(ctx);
  Machine::ProcessorQuery query(Machine::get_machine());
  query.only_kind(processor.kind());
  if(processor.id == query.first().id) {

    Legion::PhysicalRegion image = regions[0];
    ImageDescriptor imageDescriptor = ((ImageDescriptor *)task->args)[0];

    int layer = task->get_unique_id() % imageDescriptor.numImageLayers;
#if 0
    Legion::Visualization::ImageReduction::PixelField *r, *g, *b, *a, *z, *userdata;
    Legion::Visualization::ImageReduction::Stride stride;
    Legion::Visualization::ImageReduction::create_image_field_pointers(imageDescriptor, image, r, g, b, a, z, userdata, stride, runtime, ctx, true);
#else
    RWAccessor r(image, Legion::Visualization::ImageReduction::FID_FIELD_R);
    RWAccessor g(image, Legion::Visualization::ImageReduction::FID_FIELD_G);
    RWAccessor b(image, Legion::Visualization::ImageReduction::FID_FIELD_B);
    RWAccessor a(image, Legion::Visualization::ImageReduction::FID_FIELD_A);
    RWAccessor z(image, Legion::Visualization::ImageReduction::FID_FIELD_Z);
    RWAccessor userdata(image, Legion::Visualization::ImageReduction::FID_FIELD_USERDATA);
#endif
    paintRegion(imageDescriptor, r, g, b, a, z, userdata, layer);
  }
}



void top_level_task(const Legion::Task *task,
                    const std::vector<Legion::PhysicalRegion> &regions,
                    Legion::Context ctx, Legion::Runtime *runtime) {


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

  Legion::Visualization::ImageReduction imageReduction(imageDescriptor, ctx, runtime, 0);
  imageReduction.set_blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  imageReduction.set_blend_equation(GL_FUNC_ADD);

  {
    Legion::Future displayFuture;

    const int numTimeSteps = 5;

    for(int t = 0; t < numTimeSteps; ++t) {

      simulateTimeStep(t);
      Legion::FutureMap renderFutures = imageReduction.launch_task_composite_domain(
        RENDER_TASK_ID, runtime, ctx, 
        &imageDescriptor, sizeof(imageDescriptor), true);

      float cameraDirection[3] = { 0, 0, 1 };
      Legion::FutureMap reduceFutures = imageReduction.reduceImages(ctx, cameraDirection);
      reduceFutures.wait_all_results();

      displayFuture = imageReduction.display(t, ctx);
      displayFuture.wait();
    }

  }
}


static void create_mappers(Legion::Machine machine,
                           Legion::Runtime* rt,
                           const std::set<Legion::Processor>& local_procs) {
  for (Legion::Processor proc : local_procs) {
    Legion::Mapping::ImageReductionMapper* irMapper =
      new Legion::Mapping::ImageReductionMapper(rt->get_mapper_runtime(), machine, proc);
    rt->replace_default_mapper(irMapper, proc);
  }
}


#ifdef __cplusplus
extern "C" {
#endif

void register_mappers() {
  Runtime::add_registration_callback(create_mappers);
}

#ifdef __cplusplus
}
#endif




int main(const int argc, char *argv[]) {
  Legion::Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  register_mappers();

  {
    Legion::TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<top_level_task>(registrar, "top_level_task");
  }

  {
    Legion::TaskVariantRegistrar registrar(RENDER_TASK_ID, "render_task");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<render_task>(registrar, "render_task");
  }

  Legion::Visualization::ImageReduction::preinitializeBeforeRuntimeStarts();

  return Legion::Runtime::start(argc, argv);
}



