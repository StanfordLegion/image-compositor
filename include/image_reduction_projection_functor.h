
#include "legion_visualization.h"

namespace Legion {
  namespace Visualization {
    
    class ImageReductionProjectionFunctor : public ProjectionFunctor {
    public:
      ImageReductionProjectionFunctor(Domain launchDomain, LogicalPartition partition);
      
      virtual LogicalRegion project(const Mappable* mappable,
                                    unsigned index,
                                    LogicalPartition upperBound,
                                    const DomainPoint& point);
      
      virtual bool is_exclusive(void) const { return true; }
      virtual unsigned get_depth(void) const { return 0; }

#if 0
      virtual LogicalRegion project(const Mappable* mappable,
                                    unsigned index,
                                    LogicalRegion upperBound,
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
      
      virtual LogicalRegion project(const Mappable* mappable,
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
        LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(upperBound, *partitionPir);
        return result;
      }
      
      virtual LogicalRegion project(LogicalRegion upperBound,
                                    const DomainPoint& point,
                                    const Domain& launch_domain) {
        PointInRectIterator<image_region_dimensions> partitionPir(mPartitionColorSpace);
        Rect<image_region_dimensions> launchDomain = mLaunchDomain;
        
        for (PointInRectIterator<image_region_dimensions> launchPir(launchDomain); launchPir(); launchPir++) {
          if(launchPir[0] == point[0] && launchPir[1] == point[1] && launchPir[2] == point[2]) {
            break;
          }
          partitionPir++;
        }
        LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(upperBound, *partitionPir);
        return result;
      }
      
      virtual LogicalRegion project(LogicalPartition upperBound,
                                    const DomainPoint& point,
                                    const Domain& launch_domain) {
        PointInRectIterator<image_region_dimensions> partitionPir(mPartitionColorSpace);
        Rect<image_region_dimensions> launchDomain = mLaunchDomain;
        
        for (PointInRectIterator<image_region_dimensions> launchPir(launchDomain); launchPir(); launchPir++) {
          if(launchPir[0] == point[0] && launchPir[1] == point[1] && launchPir[2] == point[2]) {
            break;
          }
          partitionPir++;
        }
        LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(upperBound, *partitionPir);
        return result;
      }
      
#endif
      
      
    private:
      Domain mLaunchDomain;
      LogicalPartition mPartition;
      Domain mPartitionColorSpace;
    };
    
  }
}
