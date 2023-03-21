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

#include <unistd.h>

#include "image_reduction_mapper.h"

extern int gRenderTaskID;

namespace Legion {
  namespace Visualization {

    Logger log_image_reduction_mapper("image_reduction_mapper");
    //--------------------------------------------------------------------------
    ImageReductionMapper::ImageReductionMapper(Mapping::MapperRuntime *rt, Machine m, Processor local)
      : Mapping::DefaultMapper(rt, m, local, "image_reduction_mapper"),
        mapped_task_count(0)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    ImageReductionMapper::~ImageReductionMapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    const char* ImageReductionMapper::get_mapper_name(void) const
    //--------------------------------------------------------------------------
    {
      return (char*)"ImageReductionMapper";
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_task_options(const Mapping::MapperContext    ctx,
                                                   const Task&            task,
                                                   TaskOptions&     output)
    //--------------------------------------------------------------------------
    {
      output.initial_proc = local_proc;
      output.inline_task =  false;
      output.stealable = false;
      output.map_locally = true;
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_task(const Mapping::MapperContext      ctx,
                              const Task&              task,
                              const MapTaskInput&      input,
                                    MapTaskOutput&     output)
    //--------------------------------------------------------------------------
    {
      Processor::Kind target_kind = task.target_proc.kind();

      std::vector<VariantID> gpu_variants;
      std::vector<VariantID> cpu_variants;
      runtime->find_valid_variants(ctx, task.task_id,
                                    gpu_variants, Processor::TOC_PROC);
      runtime->find_valid_variants(ctx, task.task_id,
                                    cpu_variants, Processor::LOC_PROC);

      if (cpu_variants.empty())
        throw std::runtime_error("a task should always have a CPU variant");

      if (!local_gpus.empty() && !gpu_variants.empty()) {
        output.chosen_variant = gpu_variants[0];
        int idx = mapped_task_count++ % local_gpus.size();
        output.target_procs.push_back(local_gpus[idx]);
      }
      else {
        output.chosen_variant = cpu_variants[0];
        int idx = mapped_task_count++ % local_cpus.size();
        output.target_procs.push_back(local_cpus[idx]);
      }

      // Find the visible memories from the processor for the given kind
      Machine::MemoryQuery visible_memories(machine);
      visible_memories.best_affinity_to(output.target_procs[0]);
      if (visible_memories.count() == 0)
      {
        log_image_reduction_mapper.error("No visible memories from processor " IDFMT "! "
                         "This machine is really messed up!", task.target_proc.id);
        assert(false);
      }

      Memory target_mem = visible_memories.first(); // just take the first one
      // if (task.task_id == gRenderTaskID)
      // {
      //   for (Machine::MemoryQuery::iterator it = visible_memories.begin(); it != visible_memories.end(); it++)
      //   {
      //     if(it->kind() == Memory::Z_COPY_MEM)
      //     {
      //       target_mem = *it;
      //       break;
      //     }
      //   }
      // }
      // else
      // {
      //   target_mem = visible_memories.first(); // just take the first one
      // }

      for (size_t i = 0; i < task.regions.size(); ++i) {
        const RegionRequirement req = task.regions[i];
        Mapping::PhysicalInstance inst;

        if (req.privilege == NO_ACCESS) {
          output.chosen_instances[i].push_back(Legion::Mapping::PhysicalInstance::get_virtual_instance());
        }
        else {
          InstanceMap& imap = mem_inst_map[target_mem];
          InstanceMapKey key(req.region, req.instance_fields);
          InstanceMap::const_iterator it = imap.find(key);
          if(it != imap.end()) {
            log_image_reduction_mapper.info() << task.get_task_name() << " req: " << i
                                              << " using existing instance " << it->second
                                              << " in " << target_mem;
            inst = it->second;
            runtime->acquire_instance(ctx, inst);
          }
          else
          {
            LayoutConstraintSet constraints;
            constraints.add_constraint(FieldConstraint(req.privilege_fields,
                                                       false /*contiguous*/, false /*inorder*/));

            bool ok = runtime->create_physical_instance(
              ctx, target_mem, constraints, std::vector<LogicalRegion>{req.region}, inst);
            if(!ok)
            {
              printf("%s FAILED TO CREATE INSTANCE!!!!!\n", task.get_task_name());
              log_image_reduction_mapper.info() << task.get_task_name() << " FAILED TO CREATE INSTANCE: "
                                                << i << " " << req.region;
            }
            assert(ok);
            log_image_reduction_mapper.info() << task.get_task_name() << " new instance for req: " << i
                                              << " " << req.region << " in " << target_mem << " on "
                                              << target_mem.kind() << " " << inst;

            InstanceMap& imap = mem_inst_map[target_mem];
            imap[key] = inst;
          }
        }
        output.chosen_instances[i].push_back(inst);
      }
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::memoize_operation(const Mapping::MapperContext  ctx,
                                       const Mappable&      mappable,
                                       const MemoizeInput&  input,
                                             MemoizeOutput& output)
    //--------------------------------------------------------------------------
    {
      output.memoize = true;
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_tasks_to_map(const Mapping::MapperContext          ctx,
                                         const SelectMappingInput&    input,
                                               SelectMappingOutput&   output)
    //--------------------------------------------------------------------------
    {
      for (std::list<const Task*>::const_iterator it = input.ready_tasks.begin();
        it != input.ready_tasks.end(); it++)
        {
          output.map_tasks.insert(*it);
        }
    }

  }
}
