
#include "image_reduction_mapper.h"


ImageReductionMapper::ImageReductionMapper(MapperRuntime* rt, Machine machine, Processor local)
: DefaultMapper(rt, machine, local, "image_reduction_mapper")
{
  mRuntime = rt;
}

void ImageReductionMapper::slice_task(const MapperContext ctx,
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

void ImageReductionMapper::registerRenderTaskName(std::string name) {
  mRenderTaskNames.push_back(name);
}


Machine::ProcessorQuery ImageReductionMapper::getProcessorsFromTargetDomain(const MapperContext ctx,
                                                                            LogicalPartition partition,
                                                                            Domain::DomainPointIterator it) {
  for (Memory mem : Machine::MemoryQuery(machine)) {
    std::vector<LogicalRegion> regions;
    regions.push_back(mRuntime->get_logical_subregion_by_color(ctx, partition, it));
    LayoutConstraintSet empty_constraints;
    PhysicalInstance inst;
    if (mRuntime->find_physical_instance(ctx, mem, empty_constraints, regions, inst,
                                         false/*acquire*/,
                                         false/*tight_region_bounds*/)) {
      return Machine::ProcessorQuery(machine).has_affinity_to(mem);
    }
  }
}


void ImageReductionMapper::sliceTaskOntoProcessor(Domain domain,
                                                  Processor processor,
                                                  SliceTaskOutput& output) {
  output.slices.emplace_back(domain, processor,
                             false/*recurse*/, false/*stealable*/);
  
}


void ImageReductionMapper::sliceTaskAccordingToLogicalPartition(const MapperContext ctx,
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



