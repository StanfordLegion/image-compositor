/* Copyright 2020 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __IMAGE_REDUCTION_MAPPER_H__
#define __IMAGE_REDUCTION_MAPPER_H__

#include "legion.h"
#include "mappers/default_mapper.h"
#include <stdlib.h>
#include <assert.h>

// using namespace Legion ;
// using namespace Mapping ;

namespace Legion {
  namespace Visualization {

    /**
     * \class ImageReductionMapper
     * The null mapper provides an implementation of the mapper
     * interface that will assert on every single mapper call.
     * This is useful for building new mappers from scratch as
     * you an inherit from the null mapper and only implement
     * mapper calls that are invoked.
     */
    class ImageReductionMapper : public Mapping::DefaultMapper {
    public:
      ImageReductionMapper(Mapping::MapperRuntime *rt, Machine machine, Processor local);

      virtual ~ImageReductionMapper(void);

      virtual const char* get_mapper_name(void) const;

      virtual void select_task_options(const Mapping::MapperContext    ctx,
                                       const Task&            task,
                                             TaskOptions&     output);

      virtual void map_task(const Mapping::MapperContext      ctx,
                            const Task&              task,
                            const MapTaskInput&      input,
                                  MapTaskOutput&     output);

      virtual void memoize_operation(const Mapping::MapperContext  ctx,
                                     const Mappable&      mappable,
                                     const MemoizeInput&  input,
                                           MemoizeOutput& output);

      virtual void select_tasks_to_map(const Mapping::MapperContext          ctx,
                                       const SelectMappingInput&    input,
                                             SelectMappingOutput&   output);

    protected:
      typedef std::pair<Legion::LogicalRegion,std::vector<Legion::FieldID>> InstanceMapKey;
      typedef std::map<InstanceMapKey,Legion::Mapping::PhysicalInstance> InstanceMap;
      std::map<Memory, InstanceMap> mem_inst_map;
      unsigned long         mapped_task_count;
    };
  }; // namespace Mapping
}; // namespace Legion


#endif // __IMAGE_REDUCTION_MAPPER_H__

