//
// ImageReductionMapper.cc
//
// This mapper inherits everything from default_mapper
//
#include "image_reduction_mapper.h"

ImageReductionMapper::ImageReductionMapper(MapperRuntime* rt, Machine machine, Processor local)
: DefaultMapper(rt, machine, local, "image_reduction_mapper")
{
}
