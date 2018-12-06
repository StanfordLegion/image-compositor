
#include "image_reduction_projection_functor.h"

namespace Legion {
  namespace Visualization {
    
    ImageReductionProjectionFunctor::ImageReductionProjectionFunctor(Domain launchDomain, LogicalPartition partition) : ProjectionFunctor() {
      mLaunchDomain = launchDomain;
      mPartition = partition;
      mPartitionColorSpace = Runtime::get_runtime()->get_index_partition_color_space(partition.get_index_partition());
    }
        
    LogicalRegion ImageReductionProjectionFunctor::project(const Mappable* mappable,
                                                           unsigned index,
                                                           LogicalPartition upperBound,
                                                           const DomainPoint& point) {
      PointInRectIterator<image_region_dimensions> partitionPir(mPartitionColorSpace);
      Rect<image_region_dimensions> launchDomain = mLaunchDomain;
      
      for (PointInRectIterator<image_region_dimensions> launchPir(launchDomain); launchPir(); launchPir++) {
        if(launchPir[0] == point[0] && launchPir[1] == point[1] && launchPir[2] == point[2]) {
          break;
        }
        partitionPir++;
      }
      DomainPoint newPoint = *partitionPir;
      LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(upperBound, newPoint);
      return result;
    }
  }
  
}
