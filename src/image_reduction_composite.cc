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

#include "image_reduction_composite.h"

namespace Legion {
  namespace Visualization {

    // declare static data
    GLenum ImageReductionComposite::mGlBlendFunctionSource;
    GLenum ImageReductionComposite::mGlBlendFunctionDestination;
    GLenum ImageReductionComposite::mBlendEquation;


    inline void ImageReductionComposite::compositePixelsNever(ImageReduction::PixelField *r0,
                                                              ImageReduction::PixelField *g0,
                                                              ImageReduction::PixelField *b0,
                                                              ImageReduction::PixelField *a0,
                                                              ImageReduction::PixelField *z0,
                                                              ImageReduction::PixelField *userdata0,
                                                              ImageReduction::PixelField *r1,
                                                              ImageReduction::PixelField *g1,
                                                              ImageReduction::PixelField *b1,
                                                              ImageReduction::PixelField *a1,
                                                              ImageReduction::PixelField *z1,
                                                              ImageReduction::PixelField *userdata1,
                                                              ImageReduction::PixelField *rOut,
                                                              ImageReduction::PixelField *gOut,
                                                              ImageReduction::PixelField *bOut,
                                                              ImageReduction::PixelField *aOut,
                                                              ImageReduction::PixelField *zOut,
                                                              ImageReduction::PixelField *userdataOut,
                                                              int numPixels,
                                                              Legion::Visualization::ImageReduction::Stride stride0,
                                                              Legion::Visualization::ImageReduction::Stride stride1) {

      for(int i = 0; i < numPixels; ++i) {
        *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        increment(r0, g0, b0, a0, z0, userdata0, stride0);
        increment(r1, g1, b1, a1, z1, userdata1, stride1);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride0);
      }
    }


    /// depth composite functions


    inline void ImageReductionComposite::compositePixelsLess(ImageReduction::PixelField *r0,
                                                             ImageReduction::PixelField *g0,
                                                             ImageReduction::PixelField *b0,
                                                             ImageReduction::PixelField *a0,
                                                             ImageReduction::PixelField *z0,
                                                             ImageReduction::PixelField *userdata0,
                                                             ImageReduction::PixelField *r1,
                                                             ImageReduction::PixelField *g1,
                                                             ImageReduction::PixelField *b1,
                                                             ImageReduction::PixelField *a1,
                                                             ImageReduction::PixelField *z1,
                                                             ImageReduction::PixelField *userdata1,
                                                             ImageReduction::PixelField *rOut,
                                                             ImageReduction::PixelField *gOut,
                                                             ImageReduction::PixelField *bOut,
                                                             ImageReduction::PixelField *aOut,
                                                             ImageReduction::PixelField *zOut,
                                                             ImageReduction::PixelField *userdataOut,
                                                             int numPixels,
                                                             Legion::Visualization::ImageReduction::Stride stride0,
                                                             Legion::Visualization::ImageReduction::Stride stride1){

      for(int i = 0; i < numPixels; ++i) {
        if(*z0 < *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride0);
        increment(r1, g1, b1, a1, z1, userdata1, stride1);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride0);
      }

    }


    inline void ImageReductionComposite::compositePixelsEqual(ImageReduction::PixelField *r0,
                                                              ImageReduction::PixelField *g0,
                                                              ImageReduction::PixelField *b0,
                                                              ImageReduction::PixelField *a0,
                                                              ImageReduction::PixelField *z0,
                                                              ImageReduction::PixelField *userdata0,
                                                              ImageReduction::PixelField *r1,
                                                              ImageReduction::PixelField *g1,
                                                              ImageReduction::PixelField *b1,
                                                              ImageReduction::PixelField *a1,
                                                              ImageReduction::PixelField *z1,
                                                              ImageReduction::PixelField *userdata1,
                                                              ImageReduction::PixelField *rOut,
                                                              ImageReduction::PixelField *gOut,
                                                              ImageReduction::PixelField *bOut,
                                                              ImageReduction::PixelField *aOut,
                                                              ImageReduction::PixelField *zOut,
                                                              ImageReduction::PixelField *userdataOut,
                                                              int numPixels,
                                                              Legion::Visualization::ImageReduction::Stride stride0,
                                                              Legion::Visualization::ImageReduction::Stride stride1) {

      for(int i = 0; i < numPixels; ++i) {
        if(*z0 == *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride0);
        increment(r1, g1, b1, a1, z1, userdata1, stride1);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride0);
      }

    }


    inline void ImageReductionComposite::compositePixelsLEqual(ImageReduction::PixelField *r0,
                                                               ImageReduction::PixelField *g0,
                                                               ImageReduction::PixelField *b0,
                                                               ImageReduction::PixelField *a0,
                                                               ImageReduction::PixelField *z0,
                                                               ImageReduction::PixelField *userdata0,
                                                               ImageReduction::PixelField *r1,
                                                               ImageReduction::PixelField *g1,
                                                               ImageReduction::PixelField *b1,
                                                               ImageReduction::PixelField *a1,
                                                               ImageReduction::PixelField *z1,
                                                               ImageReduction::PixelField *userdata1,
                                                               ImageReduction::PixelField *rOut,
                                                               ImageReduction::PixelField *gOut,
                                                               ImageReduction::PixelField *bOut,
                                                               ImageReduction::PixelField *aOut,
                                                               ImageReduction::PixelField *zOut,
                                                               ImageReduction::PixelField *userdataOut,
                                                               int numPixels,
                                                               Legion::Visualization::ImageReduction::Stride stride0,
                                                               Legion::Visualization::ImageReduction::Stride stride1) {

      for(int i = 0; i < numPixels; ++i) {
        if(*z0 <= *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride0);
        increment(r1, g1, b1, a1, z1, userdata1, stride1);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride0);
      }

    }


    inline void ImageReductionComposite::compositePixelsGreater(ImageReduction::PixelField *r0,
                                                                ImageReduction::PixelField *g0,
                                                                ImageReduction::PixelField *b0,
                                                                ImageReduction::PixelField *a0,
                                                                ImageReduction::PixelField *z0,
                                                                ImageReduction::PixelField *userdata0,
                                                                ImageReduction::PixelField *r1,
                                                                ImageReduction::PixelField *g1,
                                                                ImageReduction::PixelField *b1,
                                                                ImageReduction::PixelField *a1,
                                                                ImageReduction::PixelField *z1,
                                                                ImageReduction::PixelField *userdata1,
                                                                ImageReduction::PixelField *rOut,
                                                                ImageReduction::PixelField *gOut,
                                                                ImageReduction::PixelField *bOut,
                                                                ImageReduction::PixelField *aOut,
                                                                ImageReduction::PixelField *zOut,
                                                                ImageReduction::PixelField *userdataOut,
                                                                int numPixels,
                                                                Legion::Visualization::ImageReduction::Stride stride0,
                                                                Legion::Visualization::ImageReduction::Stride stride1) {

      for(int i = 0; i < numPixels; ++i) {
        if(*z0 > *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride0);
        increment(r1, g1, b1, a1, z1, userdata1, stride1);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride0);
      }

    }

    inline void ImageReductionComposite::compositePixelsNotEqual(ImageReduction::PixelField *r0,
                                                                 ImageReduction::PixelField *g0,
                                                                 ImageReduction::PixelField *b0,
                                                                 ImageReduction::PixelField *a0,
                                                                 ImageReduction::PixelField *z0,
                                                                 ImageReduction::PixelField *userdata0,
                                                                 ImageReduction::PixelField *r1,
                                                                 ImageReduction::PixelField *g1,
                                                                 ImageReduction::PixelField *b1,
                                                                 ImageReduction::PixelField *a1,
                                                                 ImageReduction::PixelField *z1,
                                                                 ImageReduction::PixelField *userdata1,
                                                                 ImageReduction::PixelField *rOut,
                                                                 ImageReduction::PixelField *gOut,
                                                                 ImageReduction::PixelField *bOut,
                                                                 ImageReduction::PixelField *aOut,
                                                                 ImageReduction::PixelField *zOut,
                                                                 ImageReduction::PixelField *userdataOut,
                                                                 int numPixels,
                                                                 Legion::Visualization::ImageReduction::Stride stride0,
                                                                 Legion::Visualization::ImageReduction::Stride stride1) {

      for(int i = 0; i < numPixels; ++i) {
        if(*z0 != *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride0);
        increment(r1, g1, b1, a1, z1, userdata1, stride1);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride0);
      }

    }


    inline void ImageReductionComposite::compositePixelsGEqual(ImageReduction::PixelField *r0,
                                                               ImageReduction::PixelField *g0,
                                                               ImageReduction::PixelField *b0,
                                                               ImageReduction::PixelField *a0,
                                                               ImageReduction::PixelField *z0,
                                                               ImageReduction::PixelField *userdata0,
                                                               ImageReduction::PixelField *r1,
                                                               ImageReduction::PixelField *g1,
                                                               ImageReduction::PixelField *b1,
                                                               ImageReduction::PixelField *a1,
                                                               ImageReduction::PixelField *z1,
                                                               ImageReduction::PixelField *userdata1,
                                                               ImageReduction::PixelField *rOut,
                                                               ImageReduction::PixelField *gOut,
                                                               ImageReduction::PixelField *bOut,
                                                               ImageReduction::PixelField *aOut,
                                                               ImageReduction::PixelField *zOut,
                                                               ImageReduction::PixelField *userdataOut,
                                                               int numPixels,
                                                               Legion::Visualization::ImageReduction::Stride stride0,
                                                               Legion::Visualization::ImageReduction::Stride stride1) {

      for(int i = 0; i < numPixels; ++i) {
        if(*z0 >= *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride0);
        increment(r1, g1, b1, a1, z1, userdata1, stride1);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride0);
      }

    }

    inline void ImageReductionComposite::compositePixelsAlways(ImageReduction::PixelField *r0,
                                                               ImageReduction::PixelField *g0,
                                                               ImageReduction::PixelField *b0,
                                                               ImageReduction::PixelField *a0,
                                                               ImageReduction::PixelField *z0,
                                                               ImageReduction::PixelField *userdata0,
                                                               ImageReduction::PixelField *r1,
                                                               ImageReduction::PixelField *g1,
                                                               ImageReduction::PixelField *b1,
                                                               ImageReduction::PixelField *a1,
                                                               ImageReduction::PixelField *z1,
                                                               ImageReduction::PixelField *userdata1,
                                                               ImageReduction::PixelField *rOut,
                                                               ImageReduction::PixelField *gOut,
                                                               ImageReduction::PixelField *bOut,
                                                               ImageReduction::PixelField *aOut,
                                                               ImageReduction::PixelField *zOut,
                                                               ImageReduction::PixelField *userdataOut,
                                                               int numPixels,
                                                               Legion::Visualization::ImageReduction::Stride stride0,
                                                               Legion::Visualization::ImageReduction::Stride stride1) {

      // no change */
    }


    /// blending scale functions

    static inline void gl_zero(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                               ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
      factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 0;
    }



    static inline void gl_one(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                              ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
      factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 1;
    }



    static inline void gl_src_color(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                    ) {
      factors[ImageReduction::FID_FIELD_R] = *rSource;
      factors[ImageReduction::FID_FIELD_G] = *gSource;
      factors[ImageReduction::FID_FIELD_B] = *bSource;
      factors[ImageReduction::FID_FIELD_A] = *aSource;
    }



    static inline void gl_one_minus_src_color(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                              ) {
      factors[ImageReduction::FID_FIELD_R] = 1.0f - *rSource;
      factors[ImageReduction::FID_FIELD_G] = 1.0f - *gSource;
      factors[ImageReduction::FID_FIELD_B] = 1.0f - *bSource;
      factors[ImageReduction::FID_FIELD_A] = 1.0f - *aSource;

    }



    static inline void gl_dst_color(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                    ) {
      factors[ImageReduction::FID_FIELD_R] = *rDestination;
      factors[ImageReduction::FID_FIELD_G] = *gDestination;
      factors[ImageReduction::FID_FIELD_B] = *bDestination;
      factors[ImageReduction::FID_FIELD_A] = *aDestination;
    }



    static inline void gl_one_minus_dst_color(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                              ) {
      factors[ImageReduction::FID_FIELD_R] = 1.0f - *rDestination;
      factors[ImageReduction::FID_FIELD_G] = 1.0f - *gDestination;
      factors[ImageReduction::FID_FIELD_B] = 1.0f - *bDestination;
      factors[ImageReduction::FID_FIELD_A] = 1.0f - *aDestination;
    }



    static inline void gl_src_alpha(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                    ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
      factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = *aSource;
    }



    static inline void gl_one_minus_src_alpha(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                              ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
      factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 1.0f - *aSource;
    }



    static inline void gl_dst_alpha(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                    ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
      factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = *aDestination;
    }



    static inline void gl_one_minus_dst_alpha(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                              ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
      factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 1.0f - *aDestination;
    }



    static inline void gl_constant_color(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                         ) {
      factors[ImageReduction::FID_FIELD_R] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_R];
      factors[ImageReduction::FID_FIELD_G] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_G];
      factors[ImageReduction::FID_FIELD_B] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_B];
      factors[ImageReduction::FID_FIELD_A] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
    }



    static inline void gl_one_minus_constant_color(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                                   ) {
      factors[ImageReduction::FID_FIELD_R] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_R];
      factors[ImageReduction::FID_FIELD_G] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_G];
      factors[ImageReduction::FID_FIELD_B] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_B];
      factors[ImageReduction::FID_FIELD_A] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];

    }



    static inline void gl_constant_alpha(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                         ) {
      //TODO make this a local var to be consistent
      factors[ImageReduction::FID_FIELD_R] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_G] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_B] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_A] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
    }



    static inline void gl_one_minus_constant_alpha(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                                   ) {
      //TODO make this a local var to be consistent
      factors[ImageReduction::FID_FIELD_R] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_G] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_B] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_A] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
    }



    static inline void gl_src_alpha_saturate(
      ImageReduction::PixelField *rSource,
      ImageReduction::PixelField *gSource,
      ImageReduction::PixelField *bSource,
      ImageReduction::PixelField *aSource,
      ImageReduction::PixelField *rDestination,
      ImageReduction::PixelField *gDestination,
      ImageReduction::PixelField *bDestination,
      ImageReduction::PixelField *aDestination,
      ImageReduction::PixelField factors[4]
                                             ) {
      ImageReduction::PixelField i = std::min(*aSource, 1.0f - *aDestination);
      factors[ImageReduction::FID_FIELD_R] = i;
      factors[ImageReduction::FID_FIELD_G] = i;
      factors[ImageReduction::FID_FIELD_B] = i;
      factors[ImageReduction::FID_FIELD_A] = 1;
    }







    void ImageReductionComposite::callScaleFunction(GLenum blendFunction,
      ImageReduction::PixelField *r0,
                                                             ImageReduction::PixelField *g0,
                                                             ImageReduction::PixelField *b0,
                                                             ImageReduction::PixelField *a0,
                                                             ImageReduction::PixelField *r1,
                                                             ImageReduction::PixelField *g1,
                                                             ImageReduction::PixelField *b1,
                                                             ImageReduction::PixelField *a1,
                                                             ImageReduction::PixelField factors[4]
    ) {
      switch(blendFunction) {
        case GL_ZERO: gl_zero(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_ONE: gl_one(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_SRC_COLOR: gl_src_color(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_ONE_MINUS_SRC_COLOR: gl_one_minus_src_color(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_DST_COLOR: gl_dst_color(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_ONE_MINUS_DST_COLOR: gl_one_minus_dst_color(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_SRC_ALPHA: gl_src_alpha(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_ONE_MINUS_SRC_ALPHA: gl_one_minus_src_alpha(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_DST_ALPHA: gl_dst_alpha(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_ONE_MINUS_DST_ALPHA: gl_one_minus_dst_alpha(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_CONSTANT_COLOR: gl_constant_color(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_ONE_MINUS_CONSTANT_COLOR: gl_one_minus_constant_color(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_CONSTANT_ALPHA: gl_constant_alpha(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_ONE_MINUS_CONSTANT_ALPHA: gl_one_minus_constant_alpha(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
        case GL_SRC_ALPHA_SATURATE: gl_src_alpha_saturate(r0, g0, b0, a0, r1, g1, b1, a1, factors); break;
          // a few of these blend functions are not supported in our local OpenGL gl.h
          // note that we have no concept of SRC1
          // TODO this could be a problem with cross-platform compilation
          //        case GL_SRC1_COLOR: return &gl_src1_color;
          //        case GL_ONE_MINUS_SRC1_COLOR: return &gl_one_minus_src1_color;
          //        case GL_SRC1_ALPHA: return &gl_src1_alpha;
          //        case GL_ONE_MINUS_SRC1_ALPHA: return &gl_one_minus_src1_alpha;
          default: assert("unsupported value for glBlendFunctionSource or glBlendFunctionDestination");
      }

    }


    /// blend composite function for all blend operators


    // this is named "slowly" becaues it requires function calls on each pixel.
    // a fast version would require that we implement a function for each element
    //   of the cross product of blendFunctionSource X blendFunctionDestination.
    //   that is just too much coding for now, do this if performance becomes a problem.

    inline void ImageReductionComposite::blendPixelsSlowly(ImageReduction::PixelField *r0,
                                                           ImageReduction::PixelField *g0,
                                                           ImageReduction::PixelField *b0,
                                                           ImageReduction::PixelField *a0,
                                                           ImageReduction::PixelField *z0,
                                                           ImageReduction::PixelField *userdata0,
                                                           ImageReduction::PixelField *r1,
                                                           ImageReduction::PixelField *g1,
                                                           ImageReduction::PixelField *b1,
                                                           ImageReduction::PixelField *a1,
                                                           ImageReduction::PixelField *z1,
                                                           ImageReduction::PixelField *userdata1,
                                                           ImageReduction::PixelField *rOut,
                                                           ImageReduction::PixelField *gOut,
                                                           ImageReduction::PixelField *bOut,
                                                           ImageReduction::PixelField *aOut,
                                                           ImageReduction::PixelField *zOut,
                                                           ImageReduction::PixelField *userdataOut,
                                                           int numPixels,
                                                           Legion::Visualization::ImageReduction::Stride stride0,
                                                           Legion::Visualization::ImageReduction::Stride stride1) {


      for(int i = 0; i < numPixels; ++i) {

        ImageReduction::PixelField sourceFactor[4];
        callScaleFunction(mGlBlendFunctionSource, r0, g0, b0, a0, r1, g1, b1, a1, sourceFactor);
        ImageReduction::PixelField destinationFactor[4];
        callScaleFunction(mGlBlendFunctionDestination, r0, g0, b0, a0, r1, g1, b1, a1, destinationFactor);

#define SHOW_BLENDING 0
#if SHOW_BLENDING
ImageReduction::PixelField rr0 = *r0;
ImageReduction::PixelField gg0 = *g0;
ImageReduction::PixelField bb0 = *b0;
ImageReduction::PixelField aa0 = *a0;
ImageReduction::PixelField rr1 = *r1;
ImageReduction::PixelField gg1 = *g1;
ImageReduction::PixelField bb1 = *b1;
ImageReduction::PixelField aa1 = *a1;
#endif

        ImageReduction::PixelField rSource = *r0 * sourceFactor[ImageReduction::FID_FIELD_R];
        ImageReduction::PixelField gSource = *g0 * sourceFactor[ImageReduction::FID_FIELD_G];
        ImageReduction::PixelField bSource = *b0 * sourceFactor[ImageReduction::FID_FIELD_B];
        ImageReduction::PixelField aSource = *a0 * sourceFactor[ImageReduction::FID_FIELD_A];
        ImageReduction::PixelField rDestination = *r1 * destinationFactor[ImageReduction::FID_FIELD_R];
        ImageReduction::PixelField gDestination = *g1 * destinationFactor[ImageReduction::FID_FIELD_G];
        ImageReduction::PixelField bDestination = *b1 * destinationFactor[ImageReduction::FID_FIELD_B];
        ImageReduction::PixelField aDestination = *a1 * destinationFactor[ImageReduction::FID_FIELD_A];

        switch(mBlendEquation) {
          case GL_FUNC_ADD:
            *rOut = rSource + rDestination;
            *gOut = gSource + gDestination;
            *bOut = bSource + bDestination;
            *aOut = aSource + aDestination;
            break;
          case GL_FUNC_SUBTRACT:
            *rOut = rSource - rDestination;
            *gOut = gSource - gDestination;
            *bOut = bSource - bDestination;
            *aOut = aSource - aDestination;
            break;
          case GL_FUNC_REVERSE_SUBTRACT:
            *rOut = -(rSource - rDestination);
            *gOut = -(gSource - gDestination);
            *bOut = -(bSource - bDestination);
            *aOut = -(aSource - aDestination);
            break;
          case GL_MIN:
            *rOut = std::min(rSource, rDestination);
            *gOut = std::min(gSource, gDestination);
            *bOut = std::min(bSource, bDestination);
            *aOut = std::min(aSource, aDestination);
            break;
          case GL_MAX:
            *rOut = std::max(rSource, rDestination);
            *gOut = std::max(gSource, gDestination);
            *bOut = std::max(bSource, bDestination);
            *aOut = std::max(aSource, aDestination);
            break;
          default: assert(false);//should never happen
        }

        // clamp the result
        *rOut = std::min(1.0f, std::max(0.0f, *rOut));
        *gOut = std::min(1.0f, std::max(0.0f, *gOut));
        *bOut = std::min(1.0f, std::max(0.0f, *bOut));
        *aOut = std::min(1.0f, std::max(0.0f, *aOut));

#if SHOW_BLENDING
{
if(rr0 != 0 || gg0 != 0 || bb0 != 0 || rr1 != 0 || bb1 != 0 || gg1 != 0) {
  char buffer[1024];
  sprintf(buffer, "r0 %g g0 %g b0 %g a0 %g, r1 %g g1 %g b1 %g a1 %g, sourceFactor %g %g %g %g, destFactor %g %g %g %g, rSource %g rDest %g rOut %g\n",
rr0, gg0, bb0, aa0, rr1, gg1, bb1, aa1, sourceFactor[0], sourceFactor[1],
sourceFactor[2], sourceFactor[3], destinationFactor[0], destinationFactor[1],
destinationFactor[2], destinationFactor[3], rSource, rDestination, *rOut);
std::cout << buffer;
}
}
#endif

        increment(r0, g0, b0, a0, z0, userdata0, stride0);
        increment(r1, g1, b1, a1, z1, userdata1, stride1);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride0);
      }


    }


    ImageReductionComposite::CompositeFunction* ImageReductionComposite::compositeFunctionPointer(
      GLenum depthFunction, GLenum blendFunctionSource, GLenum blendFunctionDestination, GLenum blendEquation) {
      if(depthFunction != 0) {
        switch(depthFunction) {
          case GL_NEVER: return compositePixelsNever;
          case GL_LESS: return compositePixelsLess;
          case GL_EQUAL: return compositePixelsEqual;
          case GL_LEQUAL: return compositePixelsLEqual;
          case GL_GREATER: return compositePixelsGreater;
          case GL_NOTEQUAL: return compositePixelsNotEqual;
          case GL_GEQUAL: return compositePixelsGEqual;
          case GL_ALWAYS: return compositePixelsAlways;

        }
      } else {
        mGlBlendFunctionSource = blendFunctionSource;
        mGlBlendFunctionDestination = blendFunctionDestination;
        mBlendEquation = blendEquation;
        return blendPixelsSlowly;
      }
      return NULL;
    }

  }
}
