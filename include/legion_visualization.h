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


#ifndef legion_visualization_h
#define legion_visualization_h

#include "legion.h"

namespace Legion {
    namespace Visualization {
        
        static const int image_region_dimensions = 3;//(width x height) x layerID
        
        typedef struct {
            int width;
            int height;
            int numImageLayers;
            int numFragmentsPerLayer;
            
            int pixelsPerLayer() const{ return width * height; }
          
          Point<image_region_dimensions> origin() const{ return Point<image_region_dimensions>::ZEROES(); }
            Point<image_region_dimensions> upperBound() const{
                Point<image_region_dimensions> result;
                result[0] = width;
                result[1] = height;
                result[2] = numImageLayers;
                return result;
            }
            
            // launch by depth plane, each depth point is one image
            Point<image_region_dimensions> layerSize() const{
                Point<image_region_dimensions> result;
                result[0] = width;
                result[1] = height;
                result[2] = 1;
                return result;
            }
            Point<image_region_dimensions> numLayers() const{
                Point<image_region_dimensions> result;
                result[0] = 1;
                result[1] = 1;
                result[2] = numImageLayers;
                return result;
            }
            
            // launch by composite fragment,
            Point<image_region_dimensions> fragmentSize() const{
                Point<image_region_dimensions> result;
                if(numFragmentsPerLayer > height) {
                    assert((width * height) % numFragmentsPerLayer == 0);
                    result[0] = (width * height) / numFragmentsPerLayer;
                    result[1] = 1;
                    result[2] = 1;
                } else {
                    result[0] = width;
                    assert(height % numFragmentsPerLayer == 0);
                    result[1] = height / numFragmentsPerLayer;
                    result[2] = 1;
                }
                return result;
            }
            Point<image_region_dimensions> numFragments() const{
                Point<image_region_dimensions> result;
                Point<image_region_dimensions> size = fragmentSize();
                result[0] = width / size[0];
                result[1] = height / size[1];
                result[2] = numImageLayers;
                return result;
            }
                        
            Point<image_region_dimensions> incrementFragment(Point<image_region_dimensions> point) const {
                point[0] += 1;
                if(point[0] >= numFragments()[0]) {
                    point[0] = 0;
                    point[1] += 1;
                    if(point[1] >= numFragments()[1]) {
                        point[1] = 0;
                        point[2] += 1;
                        if(point[2] >= numFragments()[2]) {
                            point[2] = 0;
                        }
                    }
                }
                return point;
            }
            
            int numPixelsPerFragment() const {
                Point<image_region_dimensions> size = fragmentSize();
                int result = 1;
                for(int i = 0; i < image_region_dimensions; ++i) {
                    result *= size[i];
                }
                return result;
            }
          
          std::string toString() const {
            char buffer[512];
            sprintf(buffer, "(%dx%d) x %d layers, %d fragments per layer (%lldx%lldx%lld)",
                    width, height, numImageLayers, numFragmentsPerLayer,
                    fragmentSize()[0], fragmentSize()[1], fragmentSize()[2]);
            return std::string(buffer);
          }
          
        } ImageSize;
      
    }
}

#include "image_reduction.h"


#endif /* legion_visualization_h */
