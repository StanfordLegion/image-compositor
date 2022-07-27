//
// visualization_2_mapper.cc
//
// A simple mapper to introduce the image-reduction mapper
//

#include "legion/legion_c.h"
#include "mappers/logging_wrapper.h"
#include "mappers/default_mapper.h"
#include "visualization_3_mapper.h"
#include "image_reduction_mapper.h"
#include "render.h"

using namespace Legion;
using namespace Legion::Mapping;

class Visualization_3_Mapper : public DefaultMapper {
public:
  Visualization_3_Mapper(Runtime* rt, Machine machine, Processor local)
    : DefaultMapper(rt->get_mapper_runtime(), machine, local, "visualization_3_mapper")
    { }
};


//=============================================================================
// MAPPER REGISTRATION
//=============================================================================

static void create_mappers(Legion::Machine machine,
                           Legion::Runtime* rt,
                           const std::set<Legion::Processor>& local_procs) {
  for (Legion::Processor proc : local_procs) {
    rt->replace_default_mapper(new Visualization_3_Mapper(rt, machine, proc), proc);
    Mapping::Mapper* mapper = nullptr;
    mapper = (Mapping::Mapper*)new Legion::Visualization::ImageReductionMapper(rt->get_mapper_runtime(), machine, proc);
    mapper = new Legion::Mapping::LoggingWrapper(mapper);
    rt->add_mapper(imageReductionMapperID, mapper, proc);
  }
}

void register_mappers() {
  imageReductionMapperID = Legion::Runtime::generate_static_mapper_id();
  cxx_preinitialize();
  Runtime::add_registration_callback(create_mappers);
}

