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

namespace Legion {
  namespace Mapping {

    Logger log_image_reduction_mapper("image_reduction_mapper");

    //--------------------------------------------------------------------------
    ImageReductionMapper::ImageReductionMapper(MapperRuntime *rt, Machine m, Processor local)
      : Mapper(rt), machine(m), node_id(local.address_space()),
        local_proc(local), local_kind(local.kind()), 
        mapper_name("image_reduction_mapper")
    //--------------------------------------------------------------------------
    {

      // Get all the processors and gpus on the local node
      Machine::ProcessorQuery all_procs(machine);
      for (Machine::ProcessorQuery::iterator it = all_procs.begin();
            it != all_procs.end(); it++)
      {
        AddressSpace node = it->address_space();
        if (node == node_id)
        {
          switch (it->kind())
          {
            case Processor::TOC_PROC:
              {
                local_gpus.push_back(*it);
                break;
              }
            case Processor::LOC_PROC:
              {
                local_cpus.push_back(*it);
                break;
              }
            case Processor::IO_PROC:
              {
                local_ios.push_back(*it);
                break;
              }
            case Processor::PY_PROC:
              {
                local_pys.push_back(*it);
                break;
              }
            case Processor::PROC_SET:
              {
                local_procsets.push_back(*it);
                break;
              }
            case Processor::OMP_PROC:
              {
                local_omps.push_back(*it);
                break;
              }
            default: // ignore anything else
              break;
          }
        }
        switch (it->kind())
        {
          case Processor::TOC_PROC:
            {
              // See if we already have a target GPU processor for this node
              if (node >= remote_gpus.size())
                remote_gpus.resize(node+1, Processor::NO_PROC);
              if (!remote_gpus[node].exists())
                remote_gpus[node] = *it;
              break;
            }
          case Processor::LOC_PROC:
            {
              // See if we already have a target CPU processor for this node
              if (node >= remote_cpus.size())
                remote_cpus.resize(node+1, Processor::NO_PROC);
              if (!remote_cpus[node].exists())
                remote_cpus[node] = *it;
              break;
            }
          case Processor::IO_PROC:
            {
              // See if we already have a target I/O processor for this node
              if (node >= remote_ios.size())
                remote_ios.resize(node+1, Processor::NO_PROC);
              if (!remote_ios[node].exists())
                remote_ios[node] = *it;
              break;
            }
          case Processor::PY_PROC:
            {
              // See if we already have a target I/O processor for this node
              if (node >= remote_pys.size())
                remote_pys.resize(node+1, Processor::NO_PROC);
              if (!remote_pys[node].exists())
                remote_pys[node] = *it;
              break;
            }
          case Processor::PROC_SET:
            {
              // See if we already have a target processor set for this node
              if (node >= remote_procsets.size())
                remote_procsets.resize(node+1, Processor::NO_PROC);
              if (!remote_procsets[node].exists())
                remote_procsets[node] = *it;
              break;
            }
          case Processor::OMP_PROC:
            {
              // See if we already have a target OMP processor for this node
              if (node >= remote_omps.size())
                remote_omps.resize(node+1, Processor::NO_PROC);
              if (!remote_omps[node].exists())
                remote_omps[node] = *it;
              break;
            }
          default: // ignore anything else
            break;
        }
      }

      mapped_task_count = 0;
    }

    //--------------------------------------------------------------------------
    ImageReductionMapper::~ImageReductionMapper(void)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::report_unimplemented(const char *func_name,
                                          unsigned line) const
    //--------------------------------------------------------------------------
    {
      log_image_reduction_mapper.error("Unimplemented mapper method \"%s\" in mapper %s "
         "on line %d of %s", func_name, get_mapper_name(), line, __FILE__);
      assert(false);
    }

    //--------------------------------------------------------------------------
    const char* ImageReductionMapper::get_mapper_name(void) const    
    //--------------------------------------------------------------------------
    {
      return (char*)"ImageReductionMapper";
    }

    //--------------------------------------------------------------------------
    Mapper::MapperSyncModel ImageReductionMapper::get_mapper_sync_model(void) const
    //--------------------------------------------------------------------------
    {
      return SERIALIZED_REENTRANT_MAPPER_MODEL;
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_task_options(const MapperContext    ctx,
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
    void ImageReductionMapper::premap_task(const MapperContext      ctx,
                                 const Task&              task, 
                                 const PremapTaskInput&   input,
                                       PremapTaskOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }


    //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ Point<DIM,coord_t> ImageReductionMapper::default_select_num_blocks(
                                             long long int factor,
                                             const Rect<DIM,coord_t> &to_factor)
    //--------------------------------------------------------------------------
    {
      if (factor == 1)
      {
        Point<DIM,coord_t> ones;
        for (int i = 0; i < DIM; i++)
          ones[i] = 1;
        return ones;
      }

      // Fundamental theorem of arithmetic time!
      const unsigned num_primes = 32;
      const long long int primes[num_primes] = { 2, 3, 5, 7, 11, 13, 17, 19,
                                        23, 29, 31, 37, 41, 43, 47, 53,
                                        59, 61, 67, 71, 73, 79, 83, 89,
                                        97, 101, 103, 107, 109, 113, 127, 131 };
      // Increase the size of the prime number table if you ever hit this
      assert(factor <= (primes[num_primes-1] * primes[num_primes-1]));
      // Factor into primes
      std::vector<int> prime_factors;
      for (unsigned idx = 0; idx < num_primes; idx++)
      {
        const long long int prime = primes[idx];
        if ((prime * prime) > factor)
          break;
        while ((factor % prime) == 0)
        {
          prime_factors.push_back(prime);
          factor /= prime;
        }
        if (factor == 1)
          break;
      }
      if (factor > 1)
        prime_factors.push_back(factor);
      // Assign prime factors onto the dimensions for the target rect
      // from the largest primes down to the smallest. The goal here
      // is to assign all of the elements (in factor) while
      // maintaining a block size that is as square as possible.
      long long int result[DIM];
      for (int i = 0; i < DIM; i++)
        result[i] = 1;
      double dim_chunks[DIM];
      for (int i = 0; i < DIM; i++)
        dim_chunks[i] = ((to_factor.hi[i] - to_factor.lo[i]) + 1);
      for (int idx = prime_factors.size()-1; idx >= 0; idx--)
      {
        // Find the dimension with the biggest dim_chunk
        int next_dim = -1;
        double max_chunk = -1;
        for (int i = 0; i < DIM; i++)
        {
          if (dim_chunks[i] > max_chunk)
          {
            max_chunk = dim_chunks[i];
            next_dim = i;
          }
        }
        const long long int next_prime = prime_factors[idx];

        result[next_dim] *= next_prime;
        dim_chunks[next_dim] /= next_prime;
      }
      return Point<DIM,coord_t>(result);
    }


   //--------------------------------------------------------------------------
    template<int DIM>
    /*static*/ void ImageReductionMapper::default_decompose_points(
                           const DomainT<DIM,coord_t> &point_space,
                           const std::vector<Processor> &targets,
                           const Point<DIM,coord_t> &num_blocks,
                           bool recurse, bool stealable,
                           std::vector<TaskSlice> &slices)
    //--------------------------------------------------------------------------
    {
      Point<DIM,coord_t> zeroes;
      for (int i = 0; i < DIM; i++)
        zeroes[i] = 0;
      Point<DIM,coord_t> ones;
      for (int i = 0; i < DIM; i++)
        ones[i] = 1;
      Point<DIM,coord_t> num_points =
        point_space.bounds.hi - point_space.bounds.lo + ones;
      Rect<DIM,coord_t> blocks(zeroes, num_blocks - ones);
      size_t next_index = 0;
      slices.reserve(blocks.volume());
      for (PointInRectIterator<DIM> pir(blocks); pir(); pir++) {
        Point<DIM,coord_t> block_lo = *pir;
        Point<DIM,coord_t> block_hi = *pir + ones;
        Point<DIM,coord_t> slice_lo =
          num_points * block_lo / num_blocks + point_space.bounds.lo;
        Point<DIM,coord_t> slice_hi =
          num_points * block_hi / num_blocks + point_space.bounds.lo - ones;
        // Construct a new slice space based on the new bounds
        // and any existing sparsity map, tighten if necessary
        DomainT<DIM,coord_t> slice_space;
        slice_space.bounds.lo = slice_lo;
        slice_space.bounds.hi = slice_hi;
        slice_space.sparsity = point_space.sparsity;
        if (!slice_space.dense())
          slice_space = slice_space.tighten();
        if (slice_space.volume() > 0) {
          TaskSlice slice;
          slice.domain = slice_space;
          slice.proc = targets[next_index++ % targets.size()];
          slice.recurse = recurse;
          slice.stealable = stealable;
          slices.push_back(slice);
        }
      }
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::slice_task(const MapperContext      ctx,
                                const Task&              task, 
                                const SliceTaskInput&    input,
                                      SliceTaskOutput&   output)
    //--------------------------------------------------------------------------
    {
      bool stealing_enabled = false;

      // Figure out how many points are in this index space task
      const size_t total_points = input.domain.get_volume();

     // Do two-level slicing, first slice into slices that fit on a
      // node and then slice across the processors of the right kind
      // on the local node. If we only have one node though, just break
      // into chunks that evenly divide among processors.
      switch (input.domain.get_dim())
      {
        case 1:
          {
            DomainT<1,coord_t> point_space = input.domain;
            if (remote_cpus.size() > 1) {
              if (total_points <= local_cpus.size()) {
                Point<1,coord_t> num_blocks(local_cpus.size());
                default_decompose_points<1>(point_space, local_cpus,
                    num_blocks, false/*recurse*/,
                    stealing_enabled, output.slices);
              } else {
                Point<1,coord_t> num_blocks(remote_cpus.size());
                default_decompose_points<1>(point_space, remote_cpus,
                    num_blocks, true/*recurse*/,
                    stealing_enabled, output.slices);
              }
            } else {
              Point<1,coord_t> num_blocks(local_cpus.size());
              default_decompose_points<1>(point_space, local_cpus,
                  num_blocks, false/*recurse*/,
                  stealing_enabled, output.slices);
            }
            break;
          }
        case 2:
          {
            DomainT<2,coord_t> point_space = input.domain;
            if (remote_cpus.size() > 1) {
              if (total_points <= local_cpus.size()) {
                Point<2,coord_t> num_blocks =
                  default_select_num_blocks<2>(local_cpus.size(),point_space.bounds);
                default_decompose_points<2>(point_space, local_cpus,
                    num_blocks, false/*recurse*/,
                    stealing_enabled, output.slices);
              } else {
                Point<2,coord_t> num_blocks =
                 default_select_num_blocks<2>(remote_cpus.size(),point_space.bounds);
                default_decompose_points<2>(point_space, remote_cpus,
                    num_blocks, true/*recurse*/,
                    stealing_enabled, output.slices);
              }
            } else {
              Point<2,coord_t> num_blocks =
                default_select_num_blocks<2>(local_cpus.size(), point_space.bounds);
              default_decompose_points<2>(point_space, local_cpus,
                  num_blocks, false/*recurse*/,
                  stealing_enabled, output.slices);
            }
            break;
          }
        case 3:
          {
            DomainT<3,coord_t> point_space = input.domain;
            if (remote_cpus.size() > 1) {
              if (total_points <= local_cpus.size()) {
                Point<3,coord_t> num_blocks =
                  default_select_num_blocks<3>(local_cpus.size(),point_space.bounds);
                default_decompose_points<3>(point_space, local_cpus,
                    num_blocks, false/*recurse*/,
                    stealing_enabled, output.slices);
              } else {
                Point<3,coord_t> num_blocks =
                 default_select_num_blocks<3>(remote_cpus.size(),point_space.bounds);
                default_decompose_points<3>(point_space, remote_cpus,
                    num_blocks, true/*recurse*/,
                    stealing_enabled, output.slices);
              }
            } else {
              Point<3,coord_t> num_blocks =
               default_select_num_blocks<3>(local_cpus.size(), point_space.bounds);
              default_decompose_points<3>(point_space, local_cpus,
                  num_blocks, false/*recurse*/,
                  stealing_enabled, output.slices);
            }
            break;
          }
        default: // don't support other dimensions right now
          assert(false);
      }

    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_task(const MapperContext      ctx,
                              const Task&              task,
                              const MapTaskInput&      input,
                                    MapTaskOutput&     output)
    //--------------------------------------------------------------------------
    {
      Processor::Kind target_kind = task.target_proc.kind();

      std::vector<VariantID> variants;
      runtime->find_valid_variants(ctx, task.task_id,
                                   variants, Processor::TOC_PROC);
      if(variants.size() == 1) {
        output.chosen_variant = variants[0];
        int idx = mapped_task_count++ % local_gpus.size();
        output.target_procs.push_back(local_gpus[idx]);
      } else {
        std::vector<VariantID> variants;
        runtime->find_valid_variants(ctx, task.task_id,
                                     variants, Processor::LOC_PROC);
        if(variants.size() == 1) {
          output.chosen_variant = variants[0];
          int idx = mapped_task_count++ % local_cpus.size();
          output.target_procs.push_back(local_cpus[idx]);
        } else {
          std::vector<VariantID> variants;
          runtime->find_valid_variants(ctx, task.task_id,
                                       variants, Processor::PROC_SET);
          if(variants.size() > 0) {
            output.chosen_variant = variants[0];
            output.target_procs.push_back(task.target_proc);
          } else {
            assert(0 == "unable to find a valid task variant");
          }
        }
      }

      // Find the visible memories from the processor for the given kind
      Machine::MemoryQuery visible_memories(machine);
      visible_memories.has_affinity_to(task.target_proc);
      if (visible_memories.count() == 0)
      {
        log_image_reduction_mapper.error("No visible memories from processor " IDFMT "! "
                         "This machine is really messed up!", task.target_proc.id);
        assert(false);
      }
      Memory target_mem = visible_memories.first(); // just take the first one

      for (size_t i = 0; i < task.regions.size(); ++i) {
        const RegionRequirement req = task.regions[i];
        LayoutConstraintSet constraints;
        constraints.add_constraint(FieldConstraint(req.privilege_fields, false /*contiguous*/, false /*inorder*/));
        PhysicalInstance inst;
        bool created;
        bool ok = runtime->find_or_create_physical_instance(
          ctx, target_mem, constraints, std::vector<LogicalRegion>{req.region}, inst, created);
        assert(ok);
        output.chosen_instances[i].push_back(inst);
      }
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_replicate_task(const MapperContext      ctx,
                                        const Task&              task,
                                        const MapTaskInput&      input,
                                        const MapTaskOutput&     default_output,
                                        MapReplicateTaskOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_task_variant(const MapperContext          ctx,
                                         const Task&                  task,
                                         const SelectVariantInput&    input,
                                               SelectVariantOutput&   output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::postmap_task(const MapperContext      ctx,
                                  const Task&              task,
                                  const PostMapInput&      input,
                                        PostMapOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_task_sources(const MapperContext        ctx,
                                         const Task&                task,
                                         const SelectTaskSrcInput&  input,
                                               SelectTaskSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      default_policy_select_sources(ctx, input.target, input.source_instances,
                                    output.chosen_ranking);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::default_policy_select_sources(MapperContext ctx,
                                   const PhysicalInstance &target,
                                   const std::vector<PhysicalInstance> &sources,
                                   std::deque<PhysicalInstance> &ranking)
    //--------------------------------------------------------------------------
    {
      // For right now we'll rank instances by the bandwidth of the memory
      // they are in to the destination
      // TODO: consider layouts when ranking source  to help out the DMA system
      std::map<Memory,unsigned/*bandwidth*/> source_memories;
      Memory destination_memory = target.get_location();
      std::vector<MemoryMemoryAffinity> affinity(1);
      // fill in a vector of the sources with their bandwidths and sort them
      std::vector<std::pair<PhysicalInstance,
                          unsigned/*bandwidth*/> > band_ranking(sources.size());
      for (unsigned idx = 0; idx < sources.size(); idx++)
      {
        const PhysicalInstance &instance = sources[idx];
        Memory location = instance.get_location();
        std::map<Memory,unsigned>::const_iterator finder =
          source_memories.find(location);
        if (finder == source_memories.end())
        {
          affinity.clear();
          machine.get_mem_mem_affinity(affinity, location, destination_memory,
                                       false /*not just local affinities*/);
          unsigned memory_bandwidth = 0;
          if (!affinity.empty()) {
            assert(affinity.size() == 1);
            memory_bandwidth = affinity[0].bandwidth;
          }
          source_memories[location] = memory_bandwidth;
          band_ranking[idx] =
            std::pair<PhysicalInstance,unsigned>(instance, memory_bandwidth);
        }
        else
          band_ranking[idx] =
            std::pair<PhysicalInstance,unsigned>(instance, finder->second);
      }
      // Sort them by bandwidth
      std::sort(band_ranking.begin(), band_ranking.end(), physical_sort_func);
      // Iterate from largest bandwidth to smallest
      for (std::vector<std::pair<PhysicalInstance,unsigned> >::
            const_reverse_iterator it = band_ranking.rbegin();
            it != band_ranking.rend(); it++)
        ranking.push_back(it->first);
    }




    //--------------------------------------------------------------------------
    void ImageReductionMapper::create_task_temporary_instance(
                                    const MapperContext              ctx,
                                    const Task&                      task,
                                    const CreateTaskTemporaryInput&  input,
                                          CreateTaskTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::speculate(const MapperContext      ctx,
                               const Task&              task,
                                     SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::report_profiling(const MapperContext       ctx,
                                      const Task&               task,
                                      const TaskProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_sharding_functor(
                                 const MapperContext                ctx,
                                 const Task&                        task,
                                 const SelectShardingFunctorInput&  input,
                                       SelectShardingFunctorOutput& output)
    //--------------------------------------------------------------------------
    {
      output.chosen_functor = 0; // use the default functor
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_inline(const MapperContext        ctx,
                                const InlineMapping&       inline_op,
                                const MapInlineInput&      input,
                                      MapInlineOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_inline_sources(const MapperContext     ctx,
                                        const InlineMapping&         inline_op,
                                        const SelectInlineSrcInput&  input,
                                              SelectInlineSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::create_inline_temporary_instance(
                                  const MapperContext                ctx,
                                  const InlineMapping&               inline_op,
                                  const CreateInlineTemporaryInput&  input,
                                        CreateInlineTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);  
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::report_profiling(const MapperContext         ctx,
                                      const InlineMapping&        inline_op,
                                      const InlineProfilingInfo&  input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_copy(const MapperContext      ctx,
                              const Copy&              copy,
                              const MapCopyInput&      input,
                                    MapCopyOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_copy_sources(const MapperContext          ctx,
                                         const Copy&                  copy,
                                         const SelectCopySrcInput&    input,
                                               SelectCopySrcOutput&   output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::create_copy_temporary_instance(
                                  const MapperContext              ctx,
                                  const Copy&                      copy,
                                  const CreateCopyTemporaryInput&  input,
                                        CreateCopyTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::speculate(const MapperContext      ctx,
                               const Copy&              copy,
                                     SpeculativeOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::report_profiling(const MapperContext      ctx,
                                      const Copy&              copy,
                                      const CopyProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }
    
    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_sharding_functor(
                                 const MapperContext                ctx,
                                 const Copy&                        copy,
                                 const SelectShardingFunctorInput&  input,
                                       SelectShardingFunctorOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }
    
    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_close(const MapperContext       ctx,
                               const Close&              close,
                               const MapCloseInput&      input,
                                     MapCloseOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_close_sources(const MapperContext        ctx,
                                          const Close&               close,
                                          const SelectCloseSrcInput&  input,
                                                SelectCloseSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::create_close_temporary_instance(
                                  const MapperContext               ctx,
                                  const Close&                      close,
                                  const CreateCloseTemporaryInput&  input,
                                        CreateCloseTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::report_profiling(const MapperContext       ctx,
                                      const Close&              close,
                                      const CloseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_sharding_functor(
                                 const MapperContext                ctx,
                                 const Close&                       close,
                                 const SelectShardingFunctorInput&  input,
                                       SelectShardingFunctorOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_acquire(const MapperContext         ctx,
                                 const Acquire&              acquire,
                                 const MapAcquireInput&      input,
                                       MapAcquireOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::speculate(const MapperContext         ctx,
                               const Acquire&              acquire,
                                     SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::report_profiling(const MapperContext         ctx,
                                      const Acquire&              acquire,
                                      const AcquireProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_sharding_functor(
                                 const MapperContext                ctx,
                                 const Acquire&                     acquire,
                                 const SelectShardingFunctorInput&  input,
                                       SelectShardingFunctorOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_release(const MapperContext         ctx,
                                 const Release&              release,
                                 const MapReleaseInput&      input,
                                       MapReleaseOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_release_sources(const MapperContext      ctx,
                                        const Release&                 release,
                                        const SelectReleaseSrcInput&   input,
                                              SelectReleaseSrcOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::speculate(const MapperContext         ctx,
                               const Release&              release,
                                     SpeculativeOutput&    output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::create_release_temporary_instance(
                                   const MapperContext                 ctx,
                                   const Release&                      release,
                                   const CreateReleaseTemporaryInput&  input,
                                         CreateReleaseTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::report_profiling(const MapperContext         ctx,
                                      const Release&              release,
                                      const ReleaseProfilingInfo& input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_sharding_functor(
                                 const MapperContext                ctx,
                                 const Release&                     release,
                                 const SelectShardingFunctorInput&  input,
                                       SelectShardingFunctorOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_partition_projection(const MapperContext  ctx,
                        const Partition&                          partition,
                        const SelectPartitionProjectionInput&     input,
                              SelectPartitionProjectionOutput&    output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_partition(const MapperContext        ctx,
                               const Partition&           partition,
                               const MapPartitionInput&   input,
                                     MapPartitionOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_partition_sources(
                                     const MapperContext             ctx,
                                     const Partition&                partition,
                                     const SelectPartitionSrcInput&  input,
                                           SelectPartitionSrcOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::create_partition_temporary_instance(
                            const MapperContext                   ctx,
                            const Partition&                      partition,
                            const CreatePartitionTemporaryInput&  input,
                                  CreatePartitionTemporaryOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::report_profiling(const MapperContext              ctx,
                                    const Partition&                 partition,
                                    const PartitionProfilingInfo&    input)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_sharding_functor(
                                 const MapperContext                ctx,
                                 const Partition&                   partition,
                                 const SelectShardingFunctorInput&  input,
                                       SelectShardingFunctorOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_sharding_functor(
                                 const MapperContext                ctx,
                                 const Fill&                        fill,
                                 const SelectShardingFunctorInput&  input,
                                       SelectShardingFunctorOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::configure_context(const MapperContext         ctx,
                                       const Task&                 task,
                                             ContextConfigOutput&  output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_tunable_value(const MapperContext         ctx,
                                          const Task&                 task,
                                          const SelectTunableInput&   input,
                                                SelectTunableOutput&  output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_sharding_functor(
                                 const MapperContext                   ctx,
                                 const MustEpoch&                      epoch,
                                 const SelectShardingFunctorInput&     input,
                                       MustEpochShardingFunctorOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_must_epoch(const MapperContext           ctx,
                                    const MapMustEpochInput&      input,
                                          MapMustEpochOutput&     output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::map_dataflow_graph(const MapperContext           ctx,
                                        const MapDataflowGraphInput&  input,
                                              MapDataflowGraphOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::memoize_operation(const MapperContext  ctx,
                                       const Mappable&      mappable,
                                       const MemoizeInput&  input,
                                             MemoizeOutput& output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_tasks_to_map(const MapperContext          ctx,
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

    //--------------------------------------------------------------------------
    void ImageReductionMapper::select_steal_targets(const MapperContext         ctx,
                                          const SelectStealingInput&  input,
                                                SelectStealingOutput& output)
    //--------------------------------------------------------------------------
    {
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::permit_steal_request(const MapperContext         ctx,
                                          const StealRequestInput&    input,
                                                StealRequestOutput&   output)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::handle_message(const MapperContext           ctx,
                                    const MapperMessage&          message)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__); 
    }

    //--------------------------------------------------------------------------
    void ImageReductionMapper::handle_task_result(const MapperContext           ctx,
                                        const MapperTaskResult&       result)
    //--------------------------------------------------------------------------
    {
      report_unimplemented(__func__, __LINE__);
    }



  }; // namespace Mapping 
}; // namespace Legion
     
