//
// visualization_2_mapper.cc
//
// A simple mapper to introduce the image-reduction mapper
//

#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class Visualization_2_Mapper : public DefaultMapper {
  
}


//=============================================================================
// MAPPER REGISTRATION
//=============================================================================

static MapperID imageReductionMapperID;
class ImageReductionMapper : public DefaultMapper {
  public:
  ImageReductionMapper(MapperRuntime* rt, Machine machine, Processor local);
};

static void create_mappers(Machine machine,
                           Runtime* rt,
                           const std::set<Processor>& local_procs) {
  for (Processor proc : local_procs) {
    ImageReductionMapper* irMapper =
    new ImageReductionMapper(rt->get_mapper_runtime(), machine, proc);
    rt->add_mapper(imageReductionMapperID, (Mapping::Mapper*)irMapper, proc);
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  
  void cxx_preinitialize(MapperID);
  
#ifdef __cplusplus
}
#endif

void register_mappers() {
  imageReductionMapperID = Legion::Runtime::generate_static_mapper_id();
  cxx_preinitialize(imageReductionMapperID);
  Runtime::add_registration_callback(create_mappers);
}
