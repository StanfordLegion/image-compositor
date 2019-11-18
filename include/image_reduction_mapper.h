#include "mappers/default_mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

class ImageReductionMapper : public DefaultMapper {

public:
  ImageReductionMapper(MapperRuntime* rt, Machine machine, Processor local);
};
