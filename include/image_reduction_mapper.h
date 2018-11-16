
#include "mappers/default_mapper.h"
#include "realm/logging.h"
#include "legion_visualization.h"
#include "image_reduction_projection_functor.h"

using namespace Legion;
using namespace Legion::Mapping;
using namespace Legion::Visualization;



class ImageReductionMapper : public DefaultMapper {
  
public:
  ImageReductionMapper(MapperRuntime* rt, Machine machine, Processor local);
  
  virtual void slice_task(const MapperContext ctx,
                          const Task& task,
                          const SliceTaskInput& input,
                          SliceTaskOutput& output);
  
  static void registerRenderTaskName(std::string name);
  static bool isRenderTask(const Task& task);
  static void clearPlacement(LogicalPartition partition);

  
private:
  
  Machine::ProcessorQuery getProcessorsFromTargetDomain(const MapperContext ctx,
                                                        LogicalPartition partition,
                                                        Domain::DomainPointIterator it,
                                                        bool& result);
  
  void sliceTaskOntoProcessor(Domain domain,
                              Processor processor,
                              SliceTaskOutput& output);
  
  void sliceTaskAccordingToLogicalPartition(const MapperContext ctx,
                                            const Task& task,
                                            const SliceTaskInput& input,
                                            SliceTaskOutput& output);
  
  void sliceTaskAccordingToPreviousPartition(const MapperContext ctx,
                                             const Task& task,
                                             const SliceTaskInput& input,
                                             SliceTaskOutput& output);
private:
  MapperRuntime* mRuntime;
};
