/* Copyright 2018 Stanford University, NVIDIA Corporation
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

#include "legion.h"
#include "legion_visualization.h"
#include "image_reduction_composite.h"

#include <math.h>

namespace Legion {
  namespace Visualization {
    enum TaskIDs {
      TOP_LEVEL_TASK_ID,
      GENERATE_IMAGE_DATA_TASK_ID,
      VERIFY_COMPOSITED_IMAGE_DATA_TASK_ID,
    };
    extern const GLenum depthFuncs[];
    extern const GLenum blendFuncs[];
    extern const GLenum blendEquations[];
    extern const int numDepthFuncs;
    extern const int numBlendFuncs;
    extern const int numBlendEquations;
    
    void testAssociative(ImageReduction &imageReduction,
                         ImageSize imageSize, Context context, Runtime *runtime,
                         GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation);
    
    void testNonassociative(ImageReduction &imageReduction,
                            ImageSize imageSize, Context context, Runtime *runtime,
                            GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation);
    
    void preregisterSimulationBounds(int numSimulationBoundsX, int numSimulationBoundsY, int numSimulationBoundsZ);
    
    void generate_image_data_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime);
    
    int verify_composited_image_data_task(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context ctx, Runtime *runtime);
    
  }
}

