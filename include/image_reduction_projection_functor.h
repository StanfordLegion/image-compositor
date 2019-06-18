
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
      virtual bool is_functional(void) const { return false; }

      
    private:
      Domain mLaunchDomain;
      LogicalPartition mPartition;
      Domain mPartitionColorSpace;
    };
    
  }
}
