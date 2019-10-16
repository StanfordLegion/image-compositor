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
       ImageReduction::Stride stride0,
       ImageReduction::Stride stride1);

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

        ////DEBUG
        //        ImageReduction::PixelField* rr = r;
        //        ImageReduction::PixelField* gg = g;
        //        ImageReduction::PixelField* bb = b;
        //        ImageReduction::PixelField* aa = a;
        //        ImageReduction::PixelField* zz = z;
        //        ImageReduction::PixelField* uu = userdata;
        ////DEBUG

        r += stride[ImageReduction::FID_FIELD_R][0];
        g += stride[ImageReduction::FID_FIELD_G][0];
        b += stride[ImageReduction::FID_FIELD_B][0];
        a += stride[ImageReduction::FID_FIELD_A][0];
        z += stride[ImageReduction::FID_FIELD_Z][0];
        userdata += stride[ImageReduction::FID_FIELD_USERDATA][0];


        ////DEBUG
        //        std::cout << "increment" << std::endl;
        //        std::cout << "r " << rr << " + " << (r - rr) << " = " << r << std::endl;
        //        std::cout << "g " << gg << " + " << (g - gg) << " = " << g << std::endl;
        //        std::cout << "b " << bb << " + " << (b - bb) << " = " << b << std::endl;
        //        std::cout << "a " << aa << " + " << (a - aa) << " = " << a << std::endl;
        //        std::cout << "z " << zz << " + " << (z - zz) << " = " << z << std::endl;
        //        std::cout << "u " << uu << " + " << (userdata - uu) << " = " << userdata << std::endl;
        ////DEBUG
      }


      typedef void (*ScaleFunction)(
        ImageReduction::PixelField *rSource,
        ImageReduction::PixelField *gSource,
        ImageReduction::PixelField *bSource,
        ImageReduction::PixelField *aSource,
        ImageReduction::PixelField *rDestination,
        ImageReduction::PixelField *gDestination,
        ImageReduction::PixelField *bDestination,
        ImageReduction::PixelField *aDestination,
        ImageReduction::PixelField factors[4]
                                    );

      static void callScaleFunction(GLenum blendFunction,
        ImageReduction::PixelField *rSource,
        ImageReduction::PixelField *gSource,
        ImageReduction::PixelField *bSource,
        ImageReduction::PixelField *aSource,
        ImageReduction::PixelField *rDestination,
        ImageReduction::PixelField *gDestination,
        ImageReduction::PixelField *bDestination,
        ImageReduction::PixelField *aDestination,
        ImageReduction::PixelField factors[4]);


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
