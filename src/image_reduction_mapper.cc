
#include "mappers/default_mapper.h"
#include "realm/logging.h"
#include "legion_visualization.h"

using namespace Legion;
using namespace Legion::Mapping;
using namespace Legion::Visualization;

class ImageReductionMapper : public DefaultMapper {
  
public:
  ImageReductionMapper(MapperRuntime* rt, Machine machine, Processor local)
  : DefaultMapper(rt, machine, local, "image_compositor_mapper")
  {
    mRuntime = rt;
  }
  
  
  void registerRenderTaskName(std::string name) {
    mRenderTaskNames.push_back(name);
  }
  
  Machine::ProcessorQuery getProcessorsFromTargetDomain(const MapperContext ctx,
                                                        LogicalPartition partition,
                                                        Domain::DomainPointIterator it) {
    for (Memory mem : Machine::MemoryQuery(machine)) {
      std::vector<LogicalRegion> regions;
      regions.push_back(mRuntime->get_logical_subregion_by_color(ctx, partition, it));
      LayoutConstraintSet empty_constraints;
      PhysicalInstance inst;
      if (runtime->find_physical_instance(ctx, mem, empty_constraints, regions, inst,
                                          false/*acquire*/,
                                          false/*tight_region_bounds*/)) {
        return Machine::ProcessorQuery(machine).has_affinity_to(mem);
      }
    }
  }
  
  
  void sliceTaskOntoProcessor(Domain domain,
                              Processor processor,
                              SliceTaskOutput& output) {
    output.slices.emplace_back(domain, processor,
                               false/*recurse*/, false/*stealable*/);
    
  }
  
  
  void sliceTaskAccordingToLogicalPartition(const MapperContext ctx,
                                            const Task& task,
                                            const SliceTaskInput& input,
                                            SliceTaskOutput& output) {
    ImageDescriptor* imageDescriptor = (ImageDescriptor*)task.args;
    Domain sourceDomain = input.domain;
    Domain targetDomain = imageDescriptor->domain;
    LogicalPartition targetPartition = imageDescriptor->logicalPartition;
    Domain::DomainPointIterator targetIt(targetDomain);
    
    for(Domain::DomainPointIterator sourceIt(sourceDomain); sourceIt; sourceIt++) {
      Machine::ProcessorQuery targetProcessors = getProcessorsFromTargetDomain(ctx, targetPartition, targetIt);
      Machine::ProcessorQuery::iterator pqIt = targetProcessors.begin();
      if(pqIt != targetProcessors.end()) {
        sliceTaskOntoProcessor(sourceDomain, *pqIt, output);
      }
      targetIt++;
    }
  }
  
  
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output) {
    Domain domain = input.domain;
    assert(domain.get_dim() == 1);
    // map this 1D index task launch onto the subregions of the logical partition
    for(std::vector<std::string>::iterator it = mRenderTaskNames.begin();
        it != mRenderTaskNames.end(); ++it) {
      if(task.get_task_name() == *it) {
        sliceTaskAccordingToLogicalPartition(ctx, task, input, output);
        break;
      }
    }
  }
  
private:
  MapperRuntime* mRuntime;
  std::vector<std::string> mRenderTaskNames;
};
