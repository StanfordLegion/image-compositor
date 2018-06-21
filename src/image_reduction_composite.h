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


#ifndef ImageReductionComposite_h
#define ImageReductionComposite_h

#include "legion.h"
#include "legion_visualization.h"

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <stdio.h>

namespace Legion {
  namespace Visualization {
    
    
    class ImageReductionComposite {
    public:
      
      typedef void(CompositeFunction)
      (ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      ImageReduction::PixelField*,
      int,
      Legion::ByteOffset[ImageReduction::numPixelFields][image_region_dimensions]);
      
      static CompositeFunction* compositeFunctionPointer(GLenum depthFunction, GLenum blendFunctionSource, GLenum blendFunctionDestination, GLenum blendEquation);
      
      
      static CompositeFunction compositePixelsNever;
      static CompositeFunction compositePixelsLess;
      static CompositeFunction compositePixelsEqual;
      static CompositeFunction compositePixelsLEqual;
      static CompositeFunction compositePixelsGreater;
      static CompositeFunction compositePixelsNotEqual;
      static CompositeFunction compositePixelsGEqual;
      static CompositeFunction compositePixelsAlways;
      static CompositeFunction blendPixelsSlowly;
      
      static inline void increment(ImageReduction::PixelField *&r,
                                   ImageReduction::PixelField *&g,
                                   ImageReduction::PixelField *&b,
                                   ImageReduction::PixelField *&a,
                                   ImageReduction::PixelField *&z,
                                   ImageReduction::PixelField *&userdata,
                                   Legion::Visualization::ImageReduction::Stride stride) {
        r += stride[ImageReduction::FID_FIELD_R][0];
        g += stride[ImageReduction::FID_FIELD_G][0];
        b += stride[ImageReduction::FID_FIELD_B][0];
        a += stride[ImageReduction::FID_FIELD_A][0];
        z += stride[ImageReduction::FID_FIELD_Z][0];
        userdata += stride[ImageReduction::FID_FIELD_USERDATA][0];
      }
      
      
      typedef void (*ScaleFunction)(
      ImageReduction::PixelField *rSource0,
      ImageReduction::PixelField *gSource0,
      ImageReduction::PixelField *bSource0,
      ImageReduction::PixelField *aSource0,
      ImageReduction::PixelField *rDestination0,
      ImageReduction::PixelField *gDestination0,
      ImageReduction::PixelField *bDestination0,
      ImageReduction::PixelField *aDestination0,
      ImageReduction::PixelField factors[4]
      );
      
      static ScaleFunction getScaleFunction(GLenum blendFunction);
     
      
      // because these are static we will only be able to support one blend
      //  operation per Legion runtime instance at a time
      //  TODO remove this limitation by dynamically registering these in an array
      static ScaleFunction (mScaleFunctionSource);
      static ScaleFunction (mScaleFunctionDestination);
      static GLenum mGlBlendFunctionSource;
      static GLenum mGlBlendFunctionDestination;
      static GLenum mBlendEquation;
    };
    
  }
}


#endif /* ImageReductionComposite_h */
