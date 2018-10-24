
#include "image_reduction_mapper.h"


static std::vector<std::string> gRenderTaskNames;
static std::map<LogicalPartition, std::vector<Processor> > gPlacement;
static Realm::Logger log_mapper("image_reduction_mapper");


ImageReductionMapper::ImageReductionMapper(MapperRuntime* rt, Machine machine, Processor local)
: DefaultMapper(rt, machine, local, "image_reduction_mapper")
{
  mRuntime = rt;
}

void ImageReductionMapper::slice_task(const MapperContext ctx,
                                      const Task& task,
                                      const SliceTaskInput& input,
                                      SliceTaskOutput& output) {
  log_mapper.debug("enter image_reduction_mapper.slice_task");
  Domain domain = input.domain;
  // map this 1D index task launch onto the subregions of the logical partition
  bool found = false;
  for(std::vector<std::string>::iterator it = gRenderTaskNames.begin();
      it != gRenderTaskNames.end(); ++it) {
    if(task.get_task_name() == *it) {
      log_mapper.debug("task %s is a render task", task.get_task_name());
      sliceTaskAccordingToLogicalPartition(ctx, task, input, output);
      found = true;
      break;
    }
  }
  if(!found && task.get_task_name() == "composite_task") {
    log_mapper.debug("slicing composite_task");
    sliceTaskAccordingToPreviousPartition(ctx, task, input, output);
  } else if(!found) {
    log_mapper.debug("default slicing %s", task.get_task_name());
    return DefaultMapper::slice_task(ctx, task, input, output);
  }
  log_mapper.debug("exit image_reduction_mapper.slice_task");
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

    std::vector<LogicalRegion> regions;
    regions.push_back(mRuntime->get_logical_subregion_by_color(ctx, partition, it.p));
    LayoutConstraintSet empty_constraints;
    PhysicalInstance inst;
    if (mRuntime->find_physical_instance(ctx, mem, empty_constraints, regions, inst,
                                         false/*acquire*/,
                                         false/*tight_region_bounds*/)) {
      Machine::ProcessorQuery processorQuery = Machine::ProcessorQuery(machine).has_affinity_to(mem);
      log_mapper.debug("found physical instance with %ld processors", processorQuery.count());
      return processorQuery;
    }
  }
}


void ImageReductionMapper::sliceTaskOntoProcessor(Domain domain,
                                                  Processor processor,
                                                  SliceTaskOutput& output) {
  TaskSlice slice(domain, processor, false/*recurse*/, false/*stealable*/);
  output.slices.push_back(slice);
  
}


static const char* describeProcessor(Processor processor) {
  char buffer[128];
  unsigned pId = processor.id & 0xffffffffff;
  unsigned nodeId = (processor.id >> 40) & 0xffff;
  sprintf(buffer, "node %x proc %x", nodeId, pId);
  return std::string(buffer).c_str();
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
      log_mapper.debug("task %s sliced onto processor %s", task.get_task_name(), describeProcessor(*pqIt));
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
  
  log_mapper.debug("sourceDomain dim %d", sourceDomain.get_dim());
  for(Domain::DomainPointIterator sourceIt(sourceDomain); sourceIt; sourceIt++) {
    if(procIt != gPlacement[targetPartition].end()) {
      log_mapper.debug("task %s follows on processor %s", task.get_task_name(), describeProcessor(*procIt));
      sliceTaskOntoProcessor(sourceDomain, *procIt, output);
    }
  }
}


