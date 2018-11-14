
#include "image_reduction_mapper.h"


static std::vector<std::string> gRenderTaskNames;
typedef std::pair<Legion::Processor, Rect<3> > Placement;
static std::map<LogicalPartition, std::vector<Placement> > gPlacement;
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
log_mapper.debug("slice_task");
  log_mapper.debug("enter image_reduction_mapper.slice_task task %s", task.get_task_name());
  Domain domain = input.domain;
  // map this index task launch onto the subregions of the logical partition
  bool found = false;
  if(!strcmp(task.get_task_name(), "initial_task") || isRenderTask(task)) {
    sliceTaskAccordingToLogicalPartition(ctx, task, input, output);
  } else if(!strcmp(task.get_task_name(), "composite_task")) {
    log_mapper.debug("slicing %s", task.get_task_name());
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

bool ImageReductionMapper::isRenderTask(const Task& task) {
  for(std::vector<std::string>::iterator it = gRenderTaskNames.begin();
    it != gRenderTaskNames.end(); ++it) {
    if(!strcmp(it->c_str(), task.get_task_name())) return true;
  }
  return false;
}

void ImageReductionMapper::clearPlacement(LogicalPartition partition) {
  gPlacement[partition] = std::vector<Placement>();
}


Machine::ProcessorQuery ImageReductionMapper::getProcessorsFromTargetDomain(const MapperContext ctx,
                                                                            LogicalPartition partition,
                                                                            Domain::DomainPointIterator it,
                                                                            bool& result) {
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
                                         true/*tight_region_bounds*/)) {
      Machine::ProcessorQuery targetProcessors = Machine::ProcessorQuery(machine).same_address_space_as(mem);
      log_mapper.debug("found physical instance with %ld processors", targetProcessors.count());
      result = true;
      return targetProcessors;
    } else {
      log_mapper.debug("unable to find physical instance for subregion %lld,%lld,%lld in mem 0x%llx kind %d",
         it.p.point_data[0], it.p.point_data[1], it.p.point_data[2], mem.id, mem.kind());
    }
  }
  result = false;
}


static const char* describeProcessor(Processor processor) {
  char buffer[128];
  unsigned pId = processor.id & 0xffffffffff;
  unsigned nodeId = (processor.id >> 40) & 0xffff;
  sprintf(buffer, "node %x proc %x", nodeId, pId);
  return std::string(buffer).c_str();
}



void ImageReductionMapper::sliceTaskOntoProcessor(Domain domain,
                                                  Processor processor,
                                                  SliceTaskOutput& output) {
  TaskSlice slice(domain, processor, false/*recurse*/, false/*stealable*/);
  output.slices.push_back(slice);
  output.verify_correctness = true; 
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
  
  const int dim = 3;
  Rect<dim> rect = input.domain;
std::cout << __FUNCTION__ << " rect " << rect << task.get_task_name() << std::endl;
  for (PointInRectIterator<dim> pir(rect); pir(); pir++) {
std::cout << __FUNCTION__ << " pir " << *pir << task.get_task_name() << std::endl;
    bool ret = false;
    Machine::ProcessorQuery targetProcessors = getProcessorsFromTargetDomain(ctx, targetPartition, targetIt, ret);
    if(ret) {
      for(Machine::ProcessorQuery::iterator pqIt = targetProcessors.begin();
          pqIt != targetProcessors.end(); pqIt++) {
        if(pqIt->kind() == Processor::LOC_PROC) {
          log_mapper.debug("task %s sliced onto processor %s", task.get_task_name(), describeProcessor(*pqIt));
          Rect<dim> slice(*pir, *pir);
          sliceTaskOntoProcessor(slice, *pqIt, output);
          Placement placement = std::make_pair(*pqIt, slice);
          gPlacement[targetPartition].push_back(placement);
          break;
        }
      }
    } else {
      log_mapper.debug("task %s could not find any processors for subregion %lld,%lld,%lld",
                       task.get_task_name(),
                       targetIt.p.point_data[0], targetIt.p.point_data[1], targetIt.p.point_data[2]);
    }
    targetIt++;
  }
}


void ImageReductionMapper::sliceTaskAccordingToPreviousPartition(const MapperContext ctx,
                                                                 const Task& task,
                                                                 const SliceTaskInput& input,
                                                                 SliceTaskOutput& output) {
  ImageDescriptor* imageDescriptor = (ImageDescriptor*)task.args;
  LogicalPartition targetPartition = imageDescriptor->logicalPartition;

  for(std::vector<Placement>::iterator procIt = gPlacement[targetPartition].begin();
    procIt != gPlacement[targetPartition].end(); procIt++) {
      log_mapper.debug("task %s follows on processor %s", task.get_task_name(), describeProcessor(procIt->first));
      sliceTaskOntoProcessor(procIt->second, procIt->first, output);
  }
}


