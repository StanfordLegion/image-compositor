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

      typedef void(CompositeFunction) (
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,

        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,
        const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> >,

        int width, int height, int Z0, int Z1, bool flip
      );

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

      static CompositeFunction blendPixelsFast__ONE__ADD__ONE_MINUS_SRC_ALPHA;

      typedef void (*ScaleFunction)(
        ImageReduction::PixelField,
        ImageReduction::PixelField,
        ImageReduction::PixelField,
        ImageReduction::PixelField,
        ImageReduction::PixelField,
        ImageReduction::PixelField,
        ImageReduction::PixelField,
        ImageReduction::PixelField,
        ImageReduction::PixelField factors[4]
      );

      static void callScaleFunction(GLenum blendFunction,
        ImageReduction::PixelField rSource,
        ImageReduction::PixelField gSource,
        ImageReduction::PixelField bSource,
        ImageReduction::PixelField aSource,
        ImageReduction::PixelField rDestination,
        ImageReduction::PixelField gDestination,
        ImageReduction::PixelField bDestination,
        ImageReduction::PixelField aDestination,
        ImageReduction::PixelField factors[4]
      );

      // because these are static we will only be able to support one blend
      //  operation per Legion runtime instance at a time
      //  TODO remove this limitation by dynamically registering these in an array
      static GLenum mGlBlendFunctionSource;
      static GLenum mGlBlendFunctionDestination;
      static GLenum mBlendEquation;
    };

  }
}


#endif /* ImageReductionComposite_h */
