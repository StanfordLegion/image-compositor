//
// visualization_2_mapper.cc
//
// A simple mapper to introduce the image-reduction mapper
//

#include "legion_c.h"
#include "mappers/default_mapper.h"
#include "visualization_2_mapper.h"
#include "image_reduction_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class Visualization_2_Mapper : public DefaultMapper {
public:
  Visualization_2_Mapper(Runtime* rt, Machine machine, Processor local)
    : DefaultMapper(rt->get_mapper_runtime(), machine, local, "visualization_2_mapper")
    { }

};


//=============================================================================
// MAPPER REGISTRATION
//=============================================================================

static legion_mapper_id_t imageReductionMapperID;
static void create_mappers(Legion::Machine machine,
                           Legion::Runtime* rt,
                           const std::set<Legion::Processor>& local_procs) {
  for (Legion::Processor proc : local_procs) {
    rt->replace_default_mapper(new Visualization_2_Mapper(rt, machine, proc), proc);
    ImageReductionMapper* irMapper =
    new ImageReductionMapper(rt->get_mapper_runtime(), machine, proc);
    rt->add_mapper(imageReductionMapperID, (Mapping::Mapper*)irMapper, proc);
  }
}

#ifdef __cplusplus
extern "C" {
#endif

  void cxx_preinitialize(legion_mapper_id_t);

void register_mappers() {
  imageReductionMapperID = Legion::Runtime::generate_static_mapper_id();
  cxx_preinitialize(imageReductionMapperID);
  Runtime::add_registration_callback(create_mappers);
}

#ifdef __cplusplus
}
#endif

