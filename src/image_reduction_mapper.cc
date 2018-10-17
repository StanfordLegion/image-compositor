
#include "image_reduction_mapper.h"


static std::vector<std::string> gRenderTaskNames;
static std::map<LogicalPartition, std::vector<Processor> > gPlacement;
static Realm::Logger LOG("image_reduction_mapper");


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
  bool found = false;
  for(std::vector<std::string>::iterator it = gRenderTaskNames.begin();
      it != gRenderTaskNames.end(); ++it) {
    if(task.get_task_name() == *it) {
      LOG.debug("task %s is a render task", task.get_task_name());
      sliceTaskAccordingToLogicalPartition(ctx, task, input, output);
      found = true;
      break;
    }
  }
  if(!found && task.get_task_name() == "composite_task") {
    sliceTaskAccordingToPreviousPartition(ctx, task, input, output);
  }
}

void ImageReductionMapper::registerRenderTaskName(std::string name) {
  gRenderTaskNames.push_back(name);
}

void ImageReductionMapper::clearPlacement(LogicalPartition partition) {
  gPlacement[partition] = std::vector<Processor>();
}


Machine::ProcessorQuery ImageReductionMapper::getProcessorsFromTargetDomain(const MapperContext ctx,
                                                                            LogicalPartition partition,
                                                                            Domain::DomainPointIterator it) {
  Machine::MemoryQuery query = Machine::MemoryQuery(machine);
  for(Machine::MemoryQuery::iterator queryIt = query.begin();
      queryIt != query.end(); queryIt++) {
    Memory mem = *queryIt;
    //for (Memory mem : Machine::MemoryQuery(machine)) {
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
  TaskSlice slice(domain, processor, false/*recurse*/, false/*stealable*/);
  output.slices.push_back(slice);
  
}


static char* describeProcessor(Processor processor) {
  char buffer[128];
  unsigned pId = procId & 0xffffffffff;
  unsigned nodeId = (procId >> 40) & 0xffff;
  sprintf(buffer, "node %x proc %x", nodeId, pId);
  return buffer;
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
      LOG.debug("task %s sliced onto processor %s", task.get_task_name(), describeProcessor(*pqIt));
      sliceTaskOntoProcessor(sourceDomain, *pqIt, output);
      gPlacement[targetPartition].push_back(*pqIt);
    }
    targetIt++;
  }
}


void ImageReductionMapper::sliceTaskAccordingToPreviousPartition(const MapperContext ctx,
                                                                 const Task& task,
                                                                 const SliceTaskInput& input,
                                                                 SliceTaskOutput& output) {
  ImageDescriptor* imageDescriptor = (ImageDescriptor*)task.args;
  Domain sourceDomain = input.domain;
  LogicalPartition targetPartition = imageDescriptor->logicalPartition;
  std::vector<Processor>::iterator procIt = gPlacement[targetPartition].begin();
  
  for(Domain::DomainPointIterator sourceIt(sourceDomain); sourceIt; sourceIt++) {
    if(procIt != gPlacement[targetPartition].end()) {
      LOG.debug("task %s follows on processor %s", task.get_task_name(), describeProcessor(*procIt));
      sliceTaskOntoProcessor(sourceDomain, *procIt, output);
    }
  }
}


