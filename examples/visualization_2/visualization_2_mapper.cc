//
// visualization_2_mapper.cc
//
// A simple mapper to introduce the image-reduction mapper
//

#include "legion_c.h"
#include "mappers/default_mapper.h"
#include "visualization_2_mapper.h"

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
class ImageReductionMapper : public DefaultMapper {
  public:
  ImageReductionMapper(MapperRuntime* rt, Machine machine, Processor local);
};

static void create_mappers(Machine machine,
                           Runtime* rt,
                           const std::set<Processor>& local_procs) {
  for (Processor proc : local_procs) {
    rt->replace_default_mapper(new Visualization_2_Mapper(rt, machine, proc), proc);
#if 0
    ImageReductionMapper* irMapper =
    new ImageReductionMapper(rt->get_mapper_runtime(), machine, proc);
    rt->add_mapper(imageReductionMapperID, (Mapping::Mapper*)irMapper, proc);
#endif
  }
}

#ifdef __cplusplus
extern "C" {
#endif

  void cxx_preinitialize(legion_mapper_id_t);

#ifdef __cplusplus
}
#endif

void register_mappers() {
  imageReductionMapperID = Legion::Runtime::generate_static_mapper_id();
  cxx_preinitialize(imageReductionMapperID);
  Runtime::add_registration_callback(create_mappers);
}
