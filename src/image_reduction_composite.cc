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
    ImageReductionComposite::ScaleFunction ImageReductionComposite::mScaleFunctionSource;
    ImageReductionComposite::ScaleFunction ImageReductionComposite::mScaleFunctionDestination;
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
                                                              Legion::Visualization::ImageReduction::Stride stride) {
      
      for(int i = 0; i < numPixels; ++i) {
        *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
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
                                                             Legion::Visualization::ImageReduction::Stride stride){
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 < *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
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
                                                              Legion::Visualization::ImageReduction::Stride stride) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 == *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
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
                                                               Legion::Visualization::ImageReduction::Stride stride) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 <= *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
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
                                                                Legion::Visualization::ImageReduction::Stride stride) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 > *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
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
                                                                 Legion::Visualization::ImageReduction::Stride stride) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 != *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
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
                                                               Legion::Visualization::ImageReduction::Stride stride) {
      
      for(int i = 0; i < numPixels; ++i) {
        if(*z0 >= *z1) {
          *rOut = *r0; *gOut = *g0; *bOut = *b0; *aOut = *a0; *zOut = *z0; *userdataOut = *userdata0;
        } else {
          *rOut = *r1; *gOut = *g1; *bOut = *b1; *aOut = *a1; *zOut = *z1; *userdataOut = *userdata1;
        }
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
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
                                                               Legion::Visualization::ImageReduction::Stride stride) {
      
      // no change */
    }
    
    
    /// blending scale functions
    
    static inline void gl_zero(
                               ImageReduction::PixelField *rSource0,
                               ImageReduction::PixelField *gSource0,
                               ImageReduction::PixelField *bSource0,
                               ImageReduction::PixelField *aSource0,
                               ImageReduction::PixelField *rDestination0,
                               ImageReduction::PixelField *gDestination0,
                               ImageReduction::PixelField *bDestination0,
                               ImageReduction::PixelField *aDestination0,
                               ImageReduction::PixelField factors[4]
                               ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] = factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 0;
    }
    
    
    
    static inline void gl_one(
                              ImageReduction::PixelField *rSource0,
                              ImageReduction::PixelField *gSource0,
                              ImageReduction::PixelField *bSource0,
                              ImageReduction::PixelField *aSource0,
                              ImageReduction::PixelField *rDestination0,
                              ImageReduction::PixelField *gDestination0,
                              ImageReduction::PixelField *bDestination0,
                              ImageReduction::PixelField *aDestination0,
                              ImageReduction::PixelField factors[4]
                              ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] = factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 1;
    }
    
    
    
    static inline void gl_src_color(
                                    ImageReduction::PixelField *rSource0,
                                    ImageReduction::PixelField *gSource0,
                                    ImageReduction::PixelField *bSource0,
                                    ImageReduction::PixelField *aSource0,
                                    ImageReduction::PixelField *rDestination0,
                                    ImageReduction::PixelField *gDestination0,
                                    ImageReduction::PixelField *bDestination0,
                                    ImageReduction::PixelField *aDestination0,
                                    ImageReduction::PixelField factors[4]
                                    ) {
      factors[ImageReduction::FID_FIELD_R] = *rSource0;
      factors[ImageReduction::FID_FIELD_G] = *gSource0;
      factors[ImageReduction::FID_FIELD_B] = *bSource0;
      factors[ImageReduction::FID_FIELD_A] = *aSource0;
    }
    
    
    
    static inline void gl_one_minus_src_color(
                                              ImageReduction::PixelField *rSource0,
                                              ImageReduction::PixelField *gSource0,
                                              ImageReduction::PixelField *bSource0,
                                              ImageReduction::PixelField *aSource0,
                                              ImageReduction::PixelField *rDestination0,
                                              ImageReduction::PixelField *gDestination0,
                                              ImageReduction::PixelField *bDestination0,
                                              ImageReduction::PixelField *aDestination0,
                                              ImageReduction::PixelField factors[4]
                                              ) {
      factors[ImageReduction::FID_FIELD_R] = 1.0f - *rSource0;
      factors[ImageReduction::FID_FIELD_G] = 1.0f - *gSource0;
      factors[ImageReduction::FID_FIELD_B] = 1.0f - *bSource0;
      factors[ImageReduction::FID_FIELD_A] = 1.0f - *aSource0;
      
    }
    
    
    
    static inline void gl_dst_color(
                                    ImageReduction::PixelField *rSource0,
                                    ImageReduction::PixelField *gSource0,
                                    ImageReduction::PixelField *bSource0,
                                    ImageReduction::PixelField *aSource0,
                                    ImageReduction::PixelField *rDestination0,
                                    ImageReduction::PixelField *gDestination0,
                                    ImageReduction::PixelField *bDestination0,
                                    ImageReduction::PixelField *aDestination0,
                                    ImageReduction::PixelField factors[4]
                                    ) {
      factors[ImageReduction::FID_FIELD_R] = *rDestination0;
      factors[ImageReduction::FID_FIELD_G] = *gDestination0;
      factors[ImageReduction::FID_FIELD_B] = *bDestination0;
      factors[ImageReduction::FID_FIELD_A] = *aDestination0;
    }
    
    
    
    static inline void gl_one_minus_dst_color(
                                              ImageReduction::PixelField *rSource0,
                                              ImageReduction::PixelField *gSource0,
                                              ImageReduction::PixelField *bSource0,
                                              ImageReduction::PixelField *aSource0,
                                              ImageReduction::PixelField *rDestination0,
                                              ImageReduction::PixelField *gDestination0,
                                              ImageReduction::PixelField *bDestination0,
                                              ImageReduction::PixelField *aDestination0,
                                              ImageReduction::PixelField factors[4]
                                              ) {
      factors[ImageReduction::FID_FIELD_R] = 1.0f - *rDestination0;
      factors[ImageReduction::FID_FIELD_G] = 1.0f - *gDestination0;
      factors[ImageReduction::FID_FIELD_B] = 1.0f - *bDestination0;
      factors[ImageReduction::FID_FIELD_A] = 1.0f - *aDestination0;
    }
    
    
    
    static inline void gl_src_alpha(
                                    ImageReduction::PixelField *rSource0,
                                    ImageReduction::PixelField *gSource0,
                                    ImageReduction::PixelField *bSource0,
                                    ImageReduction::PixelField *aSource0,
                                    ImageReduction::PixelField *rDestination0,
                                    ImageReduction::PixelField *gDestination0,
                                    ImageReduction::PixelField *bDestination0,
                                    ImageReduction::PixelField *aDestination0,
                                    ImageReduction::PixelField factors[4]
                                    ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] = factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = *aSource0;
    }
    
    
    
    static inline void gl_one_minus_src_alpha(
                                              ImageReduction::PixelField *rSource0,
                                              ImageReduction::PixelField *gSource0,
                                              ImageReduction::PixelField *bSource0,
                                              ImageReduction::PixelField *aSource0,
                                              ImageReduction::PixelField *rDestination0,
                                              ImageReduction::PixelField *gDestination0,
                                              ImageReduction::PixelField *bDestination0,
                                              ImageReduction::PixelField *aDestination0,
                                              ImageReduction::PixelField factors[4]
                                              ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] = factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 1.0f - *aSource0;
    }
    
    
    
    static inline void gl_dst_alpha(
                                    ImageReduction::PixelField *rSource0,
                                    ImageReduction::PixelField *gSource0,
                                    ImageReduction::PixelField *bSource0,
                                    ImageReduction::PixelField *aSource0,
                                    ImageReduction::PixelField *rDestination0,
                                    ImageReduction::PixelField *gDestination0,
                                    ImageReduction::PixelField *bDestination0,
                                    ImageReduction::PixelField *aDestination0,
                                    ImageReduction::PixelField factors[4]
                                    ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] = factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = *aDestination0;
    }
    
    
    
    static inline void gl_one_minus_dst_alpha(
                                              ImageReduction::PixelField *rSource0,
                                              ImageReduction::PixelField *gSource0,
                                              ImageReduction::PixelField *bSource0,
                                              ImageReduction::PixelField *aSource0,
                                              ImageReduction::PixelField *rDestination0,
                                              ImageReduction::PixelField *gDestination0,
                                              ImageReduction::PixelField *bDestination0,
                                              ImageReduction::PixelField *aDestination0,
                                              ImageReduction::PixelField factors[4]
                                              ) {
      factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] = factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 1.0f - *aDestination0;
    }
    
    
    
    static inline void gl_constant_color(
                                         ImageReduction::PixelField *rSource0,
                                         ImageReduction::PixelField *gSource0,
                                         ImageReduction::PixelField *bSource0,
                                         ImageReduction::PixelField *aSource0,
                                         ImageReduction::PixelField *rDestination0,
                                         ImageReduction::PixelField *gDestination0,
                                         ImageReduction::PixelField *bDestination0,
                                         ImageReduction::PixelField *aDestination0,
                                         ImageReduction::PixelField factors[4]
                                         ) {
      factors[ImageReduction::FID_FIELD_R] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_R];
      factors[ImageReduction::FID_FIELD_G] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_G];
      factors[ImageReduction::FID_FIELD_B] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_B];
      factors[ImageReduction::FID_FIELD_A] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
    }
    
    
    
    static inline void gl_one_minus_constant_color(
                                                   ImageReduction::PixelField *rSource0,
                                                   ImageReduction::PixelField *gSource0,
                                                   ImageReduction::PixelField *bSource0,
                                                   ImageReduction::PixelField *aSource0,
                                                   ImageReduction::PixelField *rDestination0,
                                                   ImageReduction::PixelField *gDestination0,
                                                   ImageReduction::PixelField *bDestination0,
                                                   ImageReduction::PixelField *aDestination0,
                                                   ImageReduction::PixelField factors[4]
                                                   ) {
      factors[ImageReduction::FID_FIELD_R] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_R];
      factors[ImageReduction::FID_FIELD_G] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_G];
      factors[ImageReduction::FID_FIELD_B] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_B];
      factors[ImageReduction::FID_FIELD_A] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      
    }
    
    
    
    static inline void gl_constant_alpha(
                                         ImageReduction::PixelField *rSource0,
                                         ImageReduction::PixelField *gSource0,
                                         ImageReduction::PixelField *bSource0,
                                         ImageReduction::PixelField *aSource0,
                                         ImageReduction::PixelField *rDestination0,
                                         ImageReduction::PixelField *gDestination0,
                                         ImageReduction::PixelField *bDestination0,
                                         ImageReduction::PixelField *aDestination0,
                                         ImageReduction::PixelField factors[4]
                                         ) {
      //TODO make this a local var to be consistent
      factors[ImageReduction::FID_FIELD_R] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_G] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_B] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_A] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
    }
    
    
    
    static inline void gl_one_minus_constant_alpha(
                                                   ImageReduction::PixelField *rSource0,
                                                   ImageReduction::PixelField *gSource0,
                                                   ImageReduction::PixelField *bSource0,
                                                   ImageReduction::PixelField *aSource0,
                                                   ImageReduction::PixelField *rDestination0,
                                                   ImageReduction::PixelField *gDestination0,
                                                   ImageReduction::PixelField *bDestination0,
                                                   ImageReduction::PixelField *aDestination0,
                                                   ImageReduction::PixelField factors[4]
                                                   ) {
      //TODO make this a local var to be consistent
      factors[ImageReduction::FID_FIELD_R] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_G] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_B] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
      factors[ImageReduction::FID_FIELD_A] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
    }
    
    
    
    static inline void gl_src_alpha_saturate(
                                             ImageReduction::PixelField *rSource0,
                                             ImageReduction::PixelField *gSource0,
                                             ImageReduction::PixelField *bSource0,
                                             ImageReduction::PixelField *aSource0,
                                             ImageReduction::PixelField *rDestination0,
                                             ImageReduction::PixelField *gDestination0,
                                             ImageReduction::PixelField *bDestination0,
                                             ImageReduction::PixelField *aDestination0,
                                             ImageReduction::PixelField factors[4]
                                             ) {
      ImageReduction::PixelField i = std::min(*aSource0, 1.0f - *aDestination0);
      factors[ImageReduction::FID_FIELD_R] = i;
      factors[ImageReduction::FID_FIELD_G] = i;
      factors[ImageReduction::FID_FIELD_B] = i;
      factors[ImageReduction::FID_FIELD_A] = 1;
    }
    
    
    
    
    
    
    
    ImageReductionComposite::ScaleFunction ImageReductionComposite::getScaleFunction(GLenum blendFunction) {
      switch(blendFunction) {
        case GL_ZERO: return &gl_zero;
        case GL_ONE: return &gl_one;
        case GL_SRC_COLOR: return &gl_src_color;
        case GL_ONE_MINUS_SRC_COLOR: return &gl_one_minus_src_color;
        case GL_DST_COLOR: return &gl_dst_color;
        case GL_ONE_MINUS_DST_COLOR: return &gl_one_minus_dst_color;
        case GL_SRC_ALPHA: return &gl_src_alpha;
        case GL_ONE_MINUS_SRC_ALPHA: return &gl_one_minus_src_alpha;
        case GL_DST_ALPHA: return &gl_dst_alpha;
        case GL_ONE_MINUS_DST_ALPHA: return &gl_one_minus_dst_alpha;
        case GL_CONSTANT_COLOR: return &gl_constant_color;
        case GL_ONE_MINUS_CONSTANT_COLOR: return &gl_one_minus_constant_color;
        case GL_CONSTANT_ALPHA: return &gl_constant_alpha;
        case GL_ONE_MINUS_CONSTANT_ALPHA: return &gl_one_minus_constant_alpha;
        case GL_SRC_ALPHA_SATURATE: return &gl_src_alpha_saturate;
          // a few of these blend functions are not supported in our local OpenGL gl.h
          // note that we have no concept of SRC1
          // TODO this could be a problem with cross-platform compilation
          //        case GL_SRC1_COLOR: return &gl_src1_color;
          //        case GL_ONE_MINUS_SRC1_COLOR: return &gl_one_minus_src1_color;
          //        case GL_SRC1_ALPHA: return &gl_src1_alpha;
          //        case GL_ONE_MINUS_SRC1_ALPHA: return &gl_one_minus_src1_alpha;
        default: return NULL;
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
                                                           Legion::Visualization::ImageReduction::Stride stride) {
      
      for(int i = 0; i < numPixels; ++i) {
        
        ImageReduction::PixelField sourceFactor[4];
        mScaleFunctionSource(r0, g0, b0, a0, r1, g1, b1, a1, sourceFactor);
        ImageReduction::PixelField destinationFactor[4];
        mScaleFunctionDestination(r0, g0, b0, a0, r1, g1, b1, a1, destinationFactor);
        
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
        
        increment(r0, g0, b0, a0, z0, userdata0, stride);
        increment(r1, g1, b1, a1, z1, userdata1, stride);
        increment(rOut, gOut, bOut, aOut, zOut, userdataOut, stride);
      }
      
      
    }
    
    
    ImageReductionComposite::CompositeFunction* ImageReductionComposite::compositeFunctionPointer(GLenum depthFunction, GLenum blendFunctionSource, GLenum blendFunctionDestination, GLenum blendEquation) {
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
        mScaleFunctionSource = getScaleFunction(blendFunctionSource);
        mScaleFunctionDestination = getScaleFunction(blendFunctionDestination);
        if(mScaleFunctionSource != NULL && mScaleFunctionDestination != NULL) {
          mGlBlendFunctionSource = blendFunctionSource;
          mGlBlendFunctionDestination = blendFunctionDestination;
          mBlendEquation = blendEquation;
          return blendPixelsSlowly;
        }
      }
      return NULL;
    }
    
  }
}

