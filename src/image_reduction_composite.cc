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


inline void ImageReductionComposite::compositePixelsNever(const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1,
                                                          int width,
                                                          int height,
                                                          int Z0,
                                                          int Z1,
                                                          bool flip) {
  
  /*NOP*/
}


/// depth composite functions


inline void ImageReductionComposite::compositePixelsLess(const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1,
                                                         const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1,
                                                         int width,
                                                         int height,
                                                         int Z0,
                                                         int Z1,
                                                         bool flip){
  
  
  if(flip) {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] < z1[x][y][Z1]) {
          r1[x][y][Z1] = r0[x][y][Z0];
          g1[x][y][Z1] = g0[x][y][Z0];
          b1[x][y][Z1] = b0[x][y][Z0];
          a1[x][y][Z1] = a0[x][y][Z0];
          z1[x][y][Z1] = z0[x][y][Z0];
          userdata1[x][y][Z1] = userdata0[x][y][Z0];
        } else {
          /*NOP*/
        }
      }
    }
  } else {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] < z1[x][y][Z1]) {
          /*NOP*/
        } else {
          r0[x][y][Z0] = r1[x][y][Z1];
          g0[x][y][Z0] = g1[x][y][Z1];
          b0[x][y][Z0] = b1[x][y][Z1];
          a0[x][y][Z0] = a1[x][y][Z1];
          z0[x][y][Z0] = z1[x][y][Z1];
          userdata0[x][y][Z0] = userdata1[x][y][Z1];
        }
      }
    }
  }
  
}


inline void ImageReductionComposite::compositePixelsEqual(const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1,
                                                          const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1,
                                                          int width,
                                                          int height,
                                                          int Z0,
                                                          int Z1,
                                                          bool flip) {
  
  
  if(flip) {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] == z1[x][y][Z1]) {
          r1[x][y][Z1] = r0[x][y][Z0];
          g1[x][y][Z1] = g0[x][y][Z0];
          b1[x][y][Z1] = b0[x][y][Z0];
          a1[x][y][Z1] = a0[x][y][Z0];
          z1[x][y][Z1] = z0[x][y][Z0];
          userdata1[x][y][Z1] = userdata0[x][y][Z0];
        } else {
          /*NOP*/
        }
      }
    }
  } else {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] == z1[x][y][Z1]) {
          /*NOP*/
        } else {
          r0[x][y][Z0] = r1[x][y][Z1];
          g0[x][y][Z0] = g1[x][y][Z1];
          b0[x][y][Z0] = b1[x][y][Z1];
          a0[x][y][Z0] = a1[x][y][Z1];
          z0[x][y][Z0] = z1[x][y][Z1];
          userdata0[x][y][Z0] = userdata1[x][y][Z1];
        }
      }
    }
  }
  
}


inline void ImageReductionComposite::compositePixelsLEqual(const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1,
                                                           int width,
                                                           int height,
                                                           int Z0,
                                                           int Z1,
                                                           bool flip) {
  
  if(flip) {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] <= z1[x][y][Z1]) {
          r1[x][y][Z1] = r0[x][y][Z0];
          g1[x][y][Z1] = g0[x][y][Z0];
          b1[x][y][Z1] = b0[x][y][Z0];
          a1[x][y][Z1] = a0[x][y][Z0];
          z1[x][y][Z1] = z0[x][y][Z0];
          userdata1[x][y][Z1] = userdata0[x][y][Z0];
        } else {
          /*NOP*/
        }
      }
    }
  } else {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] <= z1[x][y][Z1]) {
          /*NOP*/
        } else {
          r0[x][y][Z0] = r1[x][y][Z1];
          g0[x][y][Z0] = g1[x][y][Z1];
          b0[x][y][Z0] = b1[x][y][Z1];
          a0[x][y][Z0] = a1[x][y][Z1];
          z0[x][y][Z0] = z1[x][y][Z1];
          userdata0[x][y][Z0] = userdata1[x][y][Z1];
        }
      }
    }
  }
  
  
}


inline void ImageReductionComposite::compositePixelsGreater(const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1,
                                                            const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1,
                                                            int width,
                                                            int height,
                                                            int Z0,
                                                            int Z1,
                                                            bool flip) {
  
  
  if(flip) {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] > z1[x][y][Z1]) {
          r1[x][y][Z1] = r0[x][y][Z0];
          g1[x][y][Z1] = g0[x][y][Z0];
          b1[x][y][Z1] = b0[x][y][Z0];
          a1[x][y][Z1] = a0[x][y][Z0];
          z1[x][y][Z1] = z0[x][y][Z0];
          userdata1[x][y][Z1] = userdata0[x][y][Z0];
        } else {
          /*NOP*/
        }
      }
    }
  } else {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] > z1[x][y][Z1]) {
          /*NOP*/
        } else {
          r0[x][y][Z0] = r1[x][y][Z1];
          g0[x][y][Z0] = g1[x][y][Z1];
          b0[x][y][Z0] = b1[x][y][Z1];
          a0[x][y][Z0] = a1[x][y][Z1];
          z0[x][y][Z0] = z1[x][y][Z1];
          userdata0[x][y][Z0] = userdata1[x][y][Z1];
        }
      }
    }
  }
  
}

inline void ImageReductionComposite::compositePixelsNotEqual(const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1,
                                                             const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1,
                                                             int width,
                                                             int height,
                                                             int Z0,
                                                             int Z1,
                                                             bool flip) {
  
  
  if(flip) {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] != z1[x][y][Z1]) {
          r1[x][y][Z1] = r0[x][y][Z0];
          g1[x][y][Z1] = g0[x][y][Z0];
          b1[x][y][Z1] = b0[x][y][Z0];
          a1[x][y][Z1] = a0[x][y][Z0];
          z1[x][y][Z1] = z0[x][y][Z0];
          userdata1[x][y][Z1] = userdata0[x][y][Z0];
        } else {
          /*NOP*/
        }
      }
    }
  } else {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] != z1[x][y][Z1]) {
          /*NOP*/
        } else {
          r0[x][y][Z0] = r1[x][y][Z1];
          g0[x][y][Z0] = g1[x][y][Z1];
          b0[x][y][Z0] = b1[x][y][Z1];
          a0[x][y][Z0] = a1[x][y][Z1];
          z0[x][y][Z0] = z1[x][y][Z1];
          userdata0[x][y][Z0] = userdata1[x][y][Z1];
        }
      }
    }
  }
  
}


inline void ImageReductionComposite::compositePixelsGEqual(const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1,
                                                           int width,
                                                           int height,
                                                           int Z0,
                                                           int Z1,
                                                           bool flip) {
  
  
  if(flip) {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] >= z1[x][y][Z1]) {
          r1[x][y][Z1] = r0[x][y][Z0];
          g1[x][y][Z1] = g0[x][y][Z0];
          b1[x][y][Z1] = b0[x][y][Z0];
          a1[x][y][Z1] = a0[x][y][Z0];
          z1[x][y][Z1] = z0[x][y][Z0];
          userdata1[x][y][Z1] = userdata0[x][y][Z0];
        } else {
          /*NOP*/
        }
      }
    }
  } else {
    for(int y = 0; y < height; ++y) {
      for(int x = 0; x < width; ++x) {
        if(z0[x][y][Z0] >= z1[x][y][Z1]) {
          /*NOP*/
        } else {
          r0[x][y][Z0] = r1[x][y][Z1];
          g0[x][y][Z0] = g1[x][y][Z1];
          b0[x][y][Z0] = b1[x][y][Z1];
          a0[x][y][Z0] = a1[x][y][Z1];
          z0[x][y][Z0] = z1[x][y][Z1];
          userdata0[x][y][Z0] = userdata1[x][y][Z1];
        }
      }
    }
  }
  
  
}

inline void ImageReductionComposite::compositePixelsAlways(const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1,
                                                           const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1,
                                                           int width,
                                                           int height,
                                                           int Z0,
                                                           int Z1,
                                                           bool flip) {
  
  // no change */
}


/// blending scale functions

static inline void gl_zero(
                           ImageReduction::PixelField rSource,
                           ImageReduction::PixelField gSource,
                           ImageReduction::PixelField bSource,
                           ImageReduction::PixelField aSource,
                           ImageReduction::PixelField rDestination,
                           ImageReduction::PixelField gDestination,
                           ImageReduction::PixelField bDestination,
                           ImageReduction::PixelField aDestination,
                           ImageReduction::PixelField factors[4]
                           ) {
  factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
  factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 0;
}



static inline void gl_one(
                          ImageReduction::PixelField rSource,
                          ImageReduction::PixelField gSource,
                          ImageReduction::PixelField bSource,
                          ImageReduction::PixelField aSource,
                          ImageReduction::PixelField rDestination,
                          ImageReduction::PixelField gDestination,
                          ImageReduction::PixelField bDestination,
                          ImageReduction::PixelField aDestination,
                          ImageReduction::PixelField factors[4]
                          ) {
  factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
  factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 1;
}



static inline void gl_src_color(
                                ImageReduction::PixelField rSource,
                                ImageReduction::PixelField gSource,
                                ImageReduction::PixelField bSource,
                                ImageReduction::PixelField aSource,
                                ImageReduction::PixelField rDestination,
                                ImageReduction::PixelField gDestination,
                                ImageReduction::PixelField bDestination,
                                ImageReduction::PixelField aDestination,
                                ImageReduction::PixelField factors[4]
                                ) {
  factors[ImageReduction::FID_FIELD_R] = rSource;
  factors[ImageReduction::FID_FIELD_G] = gSource;
  factors[ImageReduction::FID_FIELD_B] = bSource;
  factors[ImageReduction::FID_FIELD_A] = aSource;
}



static inline void gl_one_minus_src_color(
                                          ImageReduction::PixelField rSource,
                                          ImageReduction::PixelField gSource,
                                          ImageReduction::PixelField bSource,
                                          ImageReduction::PixelField aSource,
                                          ImageReduction::PixelField rDestination,
                                          ImageReduction::PixelField gDestination,
                                          ImageReduction::PixelField bDestination,
                                          ImageReduction::PixelField aDestination,
                                          ImageReduction::PixelField factors[4]
                                          ) {
  factors[ImageReduction::FID_FIELD_R] = 1.0f - rSource;
  factors[ImageReduction::FID_FIELD_G] = 1.0f - gSource;
  factors[ImageReduction::FID_FIELD_B] = 1.0f - bSource;
  factors[ImageReduction::FID_FIELD_A] = 1.0f - aSource;
  
}



static inline void gl_dst_color(
                                ImageReduction::PixelField rSource,
                                ImageReduction::PixelField gSource,
                                ImageReduction::PixelField bSource,
                                ImageReduction::PixelField aSource,
                                ImageReduction::PixelField rDestination,
                                ImageReduction::PixelField gDestination,
                                ImageReduction::PixelField bDestination,
                                ImageReduction::PixelField aDestination,
                                ImageReduction::PixelField factors[4]
                                ) {
  factors[ImageReduction::FID_FIELD_R] = rDestination;
  factors[ImageReduction::FID_FIELD_G] = gDestination;
  factors[ImageReduction::FID_FIELD_B] = bDestination;
  factors[ImageReduction::FID_FIELD_A] = aDestination;
}



static inline void gl_one_minus_dst_color(
                                          ImageReduction::PixelField rSource,
                                          ImageReduction::PixelField gSource,
                                          ImageReduction::PixelField bSource,
                                          ImageReduction::PixelField aSource,
                                          ImageReduction::PixelField rDestination,
                                          ImageReduction::PixelField gDestination,
                                          ImageReduction::PixelField bDestination,
                                          ImageReduction::PixelField aDestination,
                                          ImageReduction::PixelField factors[4]
                                          ) {
  factors[ImageReduction::FID_FIELD_R] = 1.0f - rDestination;
  factors[ImageReduction::FID_FIELD_G] = 1.0f - gDestination;
  factors[ImageReduction::FID_FIELD_B] = 1.0f - bDestination;
  factors[ImageReduction::FID_FIELD_A] = 1.0f - aDestination;
}



static inline void gl_src_alpha(
                                ImageReduction::PixelField rSource,
                                ImageReduction::PixelField gSource,
                                ImageReduction::PixelField bSource,
                                ImageReduction::PixelField aSource,
                                ImageReduction::PixelField rDestination,
                                ImageReduction::PixelField gDestination,
                                ImageReduction::PixelField bDestination,
                                ImageReduction::PixelField aDestination,
                                ImageReduction::PixelField factors[4]
                                ) {
  factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
  factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = aSource;
}



static inline void gl_one_minus_src_alpha(
                                          ImageReduction::PixelField rSource,
                                          ImageReduction::PixelField gSource,
                                          ImageReduction::PixelField bSource,
                                          ImageReduction::PixelField aSource,
                                          ImageReduction::PixelField rDestination,
                                          ImageReduction::PixelField gDestination,
                                          ImageReduction::PixelField bDestination,
                                          ImageReduction::PixelField aDestination,
                                          ImageReduction::PixelField factors[4]
                                          ) {
  factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
  factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 1.0f - aSource;
}



static inline void gl_dst_alpha(
                                ImageReduction::PixelField rSource,
                                ImageReduction::PixelField gSource,
                                ImageReduction::PixelField bSource,
                                ImageReduction::PixelField aSource,
                                ImageReduction::PixelField rDestination,
                                ImageReduction::PixelField gDestination,
                                ImageReduction::PixelField bDestination,
                                ImageReduction::PixelField aDestination,
                                ImageReduction::PixelField factors[4]
                                ) {
  factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
  factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = aDestination;
}



static inline void gl_one_minus_dst_alpha(
                                          ImageReduction::PixelField rSource,
                                          ImageReduction::PixelField gSource,
                                          ImageReduction::PixelField bSource,
                                          ImageReduction::PixelField aSource,
                                          ImageReduction::PixelField rDestination,
                                          ImageReduction::PixelField gDestination,
                                          ImageReduction::PixelField bDestination,
                                          ImageReduction::PixelField aDestination,
                                          ImageReduction::PixelField factors[4]
                                          ) {
  factors[ImageReduction::FID_FIELD_R] = factors[ImageReduction::FID_FIELD_G] =
  factors[ImageReduction::FID_FIELD_B] = factors[ImageReduction::FID_FIELD_A] = 1.0f - aDestination;
}



static inline void gl_constant_color(
                                     ImageReduction::PixelField rSource,
                                     ImageReduction::PixelField gSource,
                                     ImageReduction::PixelField bSource,
                                     ImageReduction::PixelField aSource,
                                     ImageReduction::PixelField rDestination,
                                     ImageReduction::PixelField gDestination,
                                     ImageReduction::PixelField bDestination,
                                     ImageReduction::PixelField aDestination,
                                     ImageReduction::PixelField factors[4]
                                     ) {
  factors[ImageReduction::FID_FIELD_R] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_R];
  factors[ImageReduction::FID_FIELD_G] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_G];
  factors[ImageReduction::FID_FIELD_B] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_B];
  factors[ImageReduction::FID_FIELD_A] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
}



static inline void gl_one_minus_constant_color(
                                               ImageReduction::PixelField rSource,
                                               ImageReduction::PixelField gSource,
                                               ImageReduction::PixelField bSource,
                                               ImageReduction::PixelField aSource,
                                               ImageReduction::PixelField rDestination,
                                               ImageReduction::PixelField gDestination,
                                               ImageReduction::PixelField bDestination,
                                               ImageReduction::PixelField aDestination,
                                               ImageReduction::PixelField factors[4]
                                               ) {
  factors[ImageReduction::FID_FIELD_R] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_R];
  factors[ImageReduction::FID_FIELD_G] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_G];
  factors[ImageReduction::FID_FIELD_B] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_B];
  factors[ImageReduction::FID_FIELD_A] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
  
}



static inline void gl_constant_alpha(
                                     ImageReduction::PixelField rSource,
                                     ImageReduction::PixelField gSource,
                                     ImageReduction::PixelField bSource,
                                     ImageReduction::PixelField aSource,
                                     ImageReduction::PixelField rDestination,
                                     ImageReduction::PixelField gDestination,
                                     ImageReduction::PixelField bDestination,
                                     ImageReduction::PixelField aDestination,
                                     ImageReduction::PixelField factors[4]
                                     ) {
  //TODO make this a local var to be consistent
  factors[ImageReduction::FID_FIELD_R] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
  factors[ImageReduction::FID_FIELD_G] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
  factors[ImageReduction::FID_FIELD_B] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
  factors[ImageReduction::FID_FIELD_A] = ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
}



static inline void gl_one_minus_constant_alpha(
                                               ImageReduction::PixelField rSource,
                                               ImageReduction::PixelField gSource,
                                               ImageReduction::PixelField bSource,
                                               ImageReduction::PixelField aSource,
                                               ImageReduction::PixelField rDestination,
                                               ImageReduction::PixelField gDestination,
                                               ImageReduction::PixelField bDestination,
                                               ImageReduction::PixelField aDestination,
                                               ImageReduction::PixelField factors[4]
                                               ) {
  //TODO make this a local var to be consistent
  factors[ImageReduction::FID_FIELD_R] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
  factors[ImageReduction::FID_FIELD_G] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
  factors[ImageReduction::FID_FIELD_B] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
  factors[ImageReduction::FID_FIELD_A] = 1.0f - ImageReduction::mGlConstantColor[ImageReduction::FID_FIELD_A];
}



static inline void gl_src_alpha_saturate(
                                         ImageReduction::PixelField rSource,
                                         ImageReduction::PixelField gSource,
                                         ImageReduction::PixelField bSource,
                                         ImageReduction::PixelField aSource,
                                         ImageReduction::PixelField rDestination,
                                         ImageReduction::PixelField gDestination,
                                         ImageReduction::PixelField bDestination,
                                         ImageReduction::PixelField aDestination,
                                         ImageReduction::PixelField factors[4]
                                         ) {
  ImageReduction::PixelField i = std::min(aSource, 1.0f - aDestination);
  factors[ImageReduction::FID_FIELD_R] = i;
  factors[ImageReduction::FID_FIELD_G] = i;
  factors[ImageReduction::FID_FIELD_B] = i;
  factors[ImageReduction::FID_FIELD_A] = 1;
}







void ImageReductionComposite::callScaleFunction(GLenum blendFunction,
                                                ImageReduction::PixelField r0,
                                                ImageReduction::PixelField g0,
                                                ImageReduction::PixelField b0,
                                                ImageReduction::PixelField a0,
                                                ImageReduction::PixelField r1,
                                                ImageReduction::PixelField g1,
                                                ImageReduction::PixelField b1,
                                                ImageReduction::PixelField a1,
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

#if 0 /* for debug purpose */
#define STACK_BUFFER(TYPE, nElements) (TYPE *)alloca(sizeof(TYPE) * nElements)
  template <typename COMP_T,
	    int N_COMP,
	    typename PIXEL_T,
	    int PIXEL_COMP,
	    bool FLIP>
  inline void writeImage(const std::string &fileName,
			 const char *const header,
			 const int sizeX,
			 const int sizeY,
			 const PIXEL_T *const pixel)
  {
    FILE *file = fopen(fileName.c_str(), "wb");
    if (file == nullptr)
      throw std::runtime_error("Can't open file for writeP[FP]M!");

    fprintf(file, header, sizeX, sizeY);
    auto out = STACK_BUFFER(COMP_T, N_COMP * sizeX);
    for (int y = 0; y < sizeY; y++) {
      auto *in = (const COMP_T *)&pixel[(FLIP ? sizeY - 1 - y : y) * sizeX];
      for (int x = 0; x < sizeX; x++)
	for (int c = 0; c < N_COMP; c++)
	  out[N_COMP * x + c] = in[PIXEL_COMP * x + (N_COMP == 1 ? 3 : c)];
      fwrite(out, N_COMP * sizeX, sizeof(COMP_T), file);
    }
    fprintf(file, "\n");
    fclose(file);
  }

  inline void writePPM(const std::string &fileName,
		       const int sizeX,
		       const int sizeY,
		       const uint32_t *pixel)
  {
    writeImage<unsigned char, 3, uint32_t, 4, true>(
						    fileName, "P6\n%i %i\n255\n", sizeX, sizeY, pixel);
  }
#endif

/// blend composite function for all blend operators


// this is named "slowly" becaues it requires function calls on each pixel.
// a fast version would require that we implement a function for each element
//   of the cross product of blendFunctionSource X blendFunctionDestination.
//   that is just too much coding for now, do this if performance becomes a problem.

inline void ImageReductionComposite::blendPixelsSlowly(const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1,
                                                       const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1,
                                                       int width,
                                                       int height,
                                                       int Z0,
                                                       int Z1,
                                                       bool flip
                                                       ) {


  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
      
      ImageReduction::PixelField sourceFactor[4];
      ImageReduction::PixelField destinationFactor[4];

      if (flip) {

        callScaleFunction(mGlBlendFunctionSource,
                          r1[x][y][Z1],
                          g1[x][y][Z1],
                          b1[x][y][Z1],
                          a1[x][y][Z1],
                          r0[x][y][Z0],
                          g0[x][y][Z0],
                          b0[x][y][Z0],
                          a0[x][y][Z0],
                          sourceFactor);
        callScaleFunction(mGlBlendFunctionDestination,
                          r1[x][y][Z1],
                          g1[x][y][Z1],
                          b1[x][y][Z1],
                          a1[x][y][Z1],
                          r0[x][y][Z0],
                          g0[x][y][Z0],
                          b0[x][y][Z0],
                          a0[x][y][Z0],
                          destinationFactor);

        ImageReduction::PixelField rSource = r1[x][y][Z1] * sourceFactor[ImageReduction::FID_FIELD_R];
        ImageReduction::PixelField gSource = g1[x][y][Z1] * sourceFactor[ImageReduction::FID_FIELD_G];
        ImageReduction::PixelField bSource = b1[x][y][Z1] * sourceFactor[ImageReduction::FID_FIELD_B];
        ImageReduction::PixelField aSource = a1[x][y][Z1] * sourceFactor[ImageReduction::FID_FIELD_A];
        ImageReduction::PixelField rDestination = r0[x][y][Z0] * destinationFactor[ImageReduction::FID_FIELD_R];
        ImageReduction::PixelField gDestination = g0[x][y][Z0] * destinationFactor[ImageReduction::FID_FIELD_G];
        ImageReduction::PixelField bDestination = b0[x][y][Z0] * destinationFactor[ImageReduction::FID_FIELD_B];
        ImageReduction::PixelField aDestination = a0[x][y][Z0] * destinationFactor[ImageReduction::FID_FIELD_A];


        switch(mBlendEquation) {
          case GL_FUNC_ADD:
            r1[x][y][Z1] = rSource + rDestination;
            g1[x][y][Z1] = gSource + gDestination;
            b1[x][y][Z1] = bSource + bDestination;
            a1[x][y][Z1] = aSource + aDestination;
            break;
          case GL_FUNC_SUBTRACT:
            r1[x][y][Z1] = rSource - rDestination;
            g1[x][y][Z1] = gSource - gDestination;
            b1[x][y][Z1] = bSource - bDestination;
            a1[x][y][Z1] = aSource - aDestination;
            break;
          case GL_FUNC_REVERSE_SUBTRACT:
            r1[x][y][Z1] = -(rSource - rDestination);
            g1[x][y][Z1] = -(gSource - gDestination);
            b1[x][y][Z1] = -(bSource - bDestination);
            a1[x][y][Z1] = -(aSource - aDestination);
            break;
          case GL_MIN:
            r1[x][y][Z1] = std::min(rSource, rDestination);
            g1[x][y][Z1] = std::min(gSource, gDestination);
            b1[x][y][Z1] = std::min(bSource, bDestination);
            a1[x][y][Z1] = std::min(aSource, aDestination);
            break;
          case GL_MAX:
            r1[x][y][Z1] = std::max(rSource, rDestination);
            g1[x][y][Z1] = std::max(gSource, gDestination);
            b1[x][y][Z1] = std::max(bSource, bDestination);
            a1[x][y][Z1] = std::max(aSource, aDestination);
            break;
          default: assert(false);//should never happen
        }
        
        // clamp the result
        r1[x][y][Z1] = std::min(1.0f, std::max(0.0f, r1[x][y][Z1]));
        g1[x][y][Z1] = std::min(1.0f, std::max(0.0f, g1[x][y][Z1]));
        b1[x][y][Z1] = std::min(1.0f, std::max(0.0f, b1[x][y][Z1]));
        a1[x][y][Z1] = std::min(1.0f, std::max(0.0f, a1[x][y][Z1]));

        r0[x][y][Z0] = r1[x][y][Z1];
        g0[x][y][Z0] = g1[x][y][Z1];
        b0[x][y][Z0] = b1[x][y][Z1];
        a0[x][y][Z0] = a1[x][y][Z1];

      } else {

        callScaleFunction(mGlBlendFunctionSource,
                          r0[x][y][Z0],
                          g0[x][y][Z0],
                          b0[x][y][Z0],
                          a0[x][y][Z0],
                          r1[x][y][Z1],
                          g1[x][y][Z1],
                          b1[x][y][Z1],
                          a1[x][y][Z1],
                          sourceFactor);
        callScaleFunction(mGlBlendFunctionDestination,
                          r0[x][y][Z0],
                          g0[x][y][Z0],
                          b0[x][y][Z0],
                          a0[x][y][Z0],
                          r1[x][y][Z1],
                          g1[x][y][Z1],
                          b1[x][y][Z1],
                          a1[x][y][Z1],
                          destinationFactor);
        
        
        ImageReduction::PixelField rSource = r0[x][y][Z0] * sourceFactor[ImageReduction::FID_FIELD_R];
        ImageReduction::PixelField gSource = g0[x][y][Z0] * sourceFactor[ImageReduction::FID_FIELD_G];
        ImageReduction::PixelField bSource = b0[x][y][Z0] * sourceFactor[ImageReduction::FID_FIELD_B];
        ImageReduction::PixelField aSource = a0[x][y][Z0] * sourceFactor[ImageReduction::FID_FIELD_A];
        ImageReduction::PixelField rDestination = r1[x][y][Z1] * destinationFactor[ImageReduction::FID_FIELD_R];
        ImageReduction::PixelField gDestination = g1[x][y][Z1] * destinationFactor[ImageReduction::FID_FIELD_G];
        ImageReduction::PixelField bDestination = b1[x][y][Z1] * destinationFactor[ImageReduction::FID_FIELD_B];
        ImageReduction::PixelField aDestination = a1[x][y][Z1] * destinationFactor[ImageReduction::FID_FIELD_A];
        
        switch(mBlendEquation) {
          case GL_FUNC_ADD:
            r0[x][y][Z0] = rSource + rDestination;
            g0[x][y][Z0] = gSource + gDestination;
            b0[x][y][Z0] = bSource + bDestination;
            a0[x][y][Z0] = aSource + aDestination;
            break;
          case GL_FUNC_SUBTRACT:
            r0[x][y][Z0] = rSource - rDestination;
            g0[x][y][Z0] = gSource - gDestination;
            b0[x][y][Z0] = bSource - bDestination;
            a0[x][y][Z0] = aSource - aDestination;
            break;
          case GL_FUNC_REVERSE_SUBTRACT:
            r0[x][y][Z0] = -(rSource - rDestination);
            g0[x][y][Z0] = -(gSource - gDestination);
            b0[x][y][Z0] = -(bSource - bDestination);
            a0[x][y][Z0] = -(aSource - aDestination);
            break;
          case GL_MIN:
            r0[x][y][Z0] = std::min(rSource, rDestination);
            g0[x][y][Z0] = std::min(gSource, gDestination);
            b0[x][y][Z0] = std::min(bSource, bDestination);
            a0[x][y][Z0] = std::min(aSource, aDestination);
            break;
          case GL_MAX:
            r0[x][y][Z0] = std::max(rSource, rDestination);
            g0[x][y][Z0] = std::max(gSource, gDestination);
            b0[x][y][Z0] = std::max(bSource, bDestination);
            a0[x][y][Z0] = std::max(aSource, aDestination);
            break;
          default: assert(false);//should never happen
        }
        
        // clamp the result
        r0[x][y][Z0] = std::min(1.0f, std::max(0.0f, r0[x][y][Z0]));
        g0[x][y][Z0] = std::min(1.0f, std::max(0.0f, g0[x][y][Z0]));
        b0[x][y][Z0] = std::min(1.0f, std::max(0.0f, b0[x][y][Z0]));
        a0[x][y][Z0] = std::min(1.0f, std::max(0.0f, a0[x][y][Z0]));

      }
      
    }
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

