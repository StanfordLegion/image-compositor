/* Copyright 2017 Stanford University, NVIDIA Corporation
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


#include "legion_visualization.h"
#include "image_reduction_composite.h"

#include "mappers/default_mapper.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


namespace Legion {
  namespace Visualization {
    
    
    // declare module static data
    
    int ImageReduction::mNodeID;
    ImageReduction::SimulationBoundsCoordinate *ImageReduction::mSimulationBounds;
    int ImageReduction::mNumSimulationBounds;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mXMax;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mXMin;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mYMax;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mYMin;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mZMax;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mZMin;
    std::vector<ImageReduction::CompositeProjectionFunctor*> *ImageReduction::mCompositeProjectionFunctor = NULL;
    std::vector<Domain> *ImageReduction::mHierarchicalTreeDomain = NULL;
    GLfloat ImageReduction::mGlViewTransform[numMatrixElements4x4];
    ImageReduction::PixelField ImageReduction::mGlConstantColor[numPixelFields];
    GLenum ImageReduction::mGlBlendEquation;
    GLenum ImageReduction::mGlBlendFunctionSource;
    GLenum ImageReduction::mGlBlendFunctionDestination;
    TaskID ImageReduction::mInitialTaskID;
    TaskID ImageReduction::mCompositeTaskID;
    TaskID ImageReduction::mDisplayTaskID;
    MapperID gMapperID;
    
    
    ImageReduction::ImageReduction(LogicalPartition partition, ImageDescriptor imageDescriptor, Context context, HighLevelRuntime *runtime, MapperID mapperID) {
      Domain domain = runtime->get_index_partition_color_space(context, partition.get_index_partition());
      imageDescriptor.logicalPartition = partition;
      imageDescriptor.domain = domain;
      imageDescriptor.numImageLayers = domain.get_volume();
      std::cout << __FUNCTION__ << " domain " << domain << std::endl;
      mImageDescriptor = imageDescriptor;
      mRuntime = runtime;
      mDepthFunction = 0;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mNodeID = -1;
      mMapperID = mapperID;
      gMapperID = mapperID;
      
      mGlBlendEquation = GL_FUNC_ADD;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mDepthFunction = 0;
      legion_field_id_t fieldID[6];

      createImage(mSourceIndexSpace, mSourceImage, mSourceImageDomain, mSourceImageFields, fieldID, context);
      partitionImageEverywhere(mSourceImage, mEverywhereDomain, mEverywherePartition, context, runtime, imageDescriptor);
      
      initializeNodes(runtime, context);
      
      assert(mNodeID != -1);
      initializeViewMatrix();
      createTreeDomains(mNodeID, numTreeLevels(imageDescriptor), runtime, imageDescriptor);
      
    }
    
    ImageReduction::ImageReduction(ImageDescriptor imageDescriptor, Context context, HighLevelRuntime *runtime, MapperID mapperID) {
      mImageDescriptor = imageDescriptor;
      mRuntime = runtime;
      mDepthFunction = 0;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mNodeID = -1;
      mMapperID = mapperID;
      gMapperID = mapperID;
      
      mGlBlendEquation = GL_FUNC_ADD;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mDepthFunction = 0;
      legion_field_id_t fieldID[6];

      createImage(mSourceIndexSpace, mSourceImage, mSourceImageDomain, mSourceImageFields, fieldID, context);
      partitionImageEverywhere(mSourceImage, mEverywhereDomain, mEverywherePartition, context, runtime, imageDescriptor);

      initializeNodes(runtime, context);

      assert(mNodeID != -1);
      initializeViewMatrix();
      createTreeDomains(mNodeID, numTreeLevels(imageDescriptor), runtime, imageDescriptor);
    }
    
    ImageReduction::~ImageReduction() {
      if(mHierarchicalTreeDomain != NULL) {
        delete mHierarchicalTreeDomain;
        mHierarchicalTreeDomain = NULL;
      }
      if(mCompositeProjectionFunctor != NULL) {
        delete mCompositeProjectionFunctor;
        mCompositeProjectionFunctor = NULL;
      }
      
#if 0 // can't destroy these, missing context
      mRuntime->destroy_index_space(context, mSourceImage.get_index_space());
      mRuntime->destroy_logical_region(context, mSourceImage);
      mRuntime->destroy_index_partition(context, mDepthPartition.get_index_partition());
      mRuntime->destroy_logical_partition(context, mDepthPartition);
      mRuntime->destroy_index_partition(context, mSourceFragmentPartition.get_index_partition());
      mRuntime->destroy_logical_partition(context, mSourceFragmentPartition);
#endif
    }
    
    
    // this function should always be called prior to starting the Legion runtime
    
    void ImageReduction::preinitializeBeforeRuntimeStarts() {
      registerTasks();
    }
    
    
    // this function should be called prior to starting the Legion runtime if you
    // plan to use noncommutative reductions
    // its purpose is to copy the domain bounds to all nodes
    
    void ImageReduction::preregisterSimulationBounds(SimulationBoundsCoordinate *bounds, int numBounds) {
      mNumSimulationBounds = numBounds;
      
      int totalElements = numBounds * fieldsPerSimulationBounds;
      if(mSimulationBounds != NULL) {
        delete [] mSimulationBounds;
      }
      mSimulationBounds = new SimulationBoundsCoordinate[totalElements];
      memcpy(mSimulationBounds, bounds, sizeof(SimulationBoundsCoordinate) * totalElements);
      mXMin = mYMin = mZMin = 1.0e+32;
      mXMax = mYMax = mZMax = -mXMin;
      SimulationBoundsCoordinate *bound = mSimulationBounds;
      for(int i = 0; i < numBounds; ++i) {
        mXMin = (bound[0] < mXMin) ? bound[0] : mXMin;
        mYMin = (bound[1] < mYMin) ? bound[1] : mYMin;
        mZMin = (bound[2] < mZMin) ? bound[2] : mZMin;
        mXMax = (bound[3] > mXMax) ? bound[3] : mXMax;
        mYMax = (bound[4] > mYMax) ? bound[4] : mYMax;
        mZMax = (bound[5] > mZMax) ? bound[5] : mZMax;
        bound += fieldsPerSimulationBounds;
      }
      
    }
    
    // this function should be called prior to starting the Legion runtime
    // its purpose is to register tasks with the same id on all nodes
    
    void ImageReduction::registerTasks() {
      
      {
        mInitialTaskID = Legion::HighLevelRuntime::generate_static_task_id();
        TaskVariantRegistrar registrar(mInitialTaskID, "initial_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<initial_task>(registrar, "initial_task");
      }
      {
        mCompositeTaskID = Legion::HighLevelRuntime::generate_static_task_id();
        TaskVariantRegistrar registrar(mCompositeTaskID, "composite_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<composite_task>(registrar, "composite_task");
      }
      {
        mDisplayTaskID = Legion::HighLevelRuntime::generate_static_task_id();
        TaskVariantRegistrar registrar(mDisplayTaskID, "display_task");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<display_task>(registrar, "display_task");
      }
    }
    
    
    FieldSpace ImageReduction::imageFields(Context context) {
      FieldSpace fields = mRuntime->create_field_space(context);
      mRuntime->attach_name(fields, "pixel fields");
      {
        FieldAllocator allocator = mRuntime->create_field_allocator(context, fields);
        FieldID fidr = allocator.allocate_field(sizeof(PixelField), FID_FIELD_R);
        assert(fidr == FID_FIELD_R);
        FieldID fidg = allocator.allocate_field(sizeof(PixelField), FID_FIELD_G);
        assert(fidg == FID_FIELD_G);
        FieldID fidb = allocator.allocate_field(sizeof(PixelField), FID_FIELD_B);
        assert(fidb == FID_FIELD_B);
        FieldID fida = allocator.allocate_field(sizeof(PixelField), FID_FIELD_A);
        assert(fida == FID_FIELD_A);
        FieldID fidz = allocator.allocate_field(sizeof(PixelField), FID_FIELD_Z);
        assert(fidz == FID_FIELD_Z);
        FieldID fidUserdata = allocator.allocate_field(sizeof(PixelField), FID_FIELD_USERDATA);
        assert(fidUserdata == FID_FIELD_USERDATA);
      }
      return fields;
    }
    
    
    void ImageReduction::createImage(IndexSpace& indexSpace, LogicalRegion &region, Domain &domain, FieldSpace& fields, legion_field_id_t fieldID[], Context context) {
      Point<image_region_dimensions> p0;
      p0 = mImageDescriptor.origin();
      Point <image_region_dimensions> p1;
      p1 = mImageDescriptor.upperBound();
      Point <image_region_dimensions> p2(1);
      p2 = Point<image_region_dimensions>(1);
      p2 = mImageDescriptor.upperBound() - Point<image_region_dimensions>(1);
      Point <image_region_dimensions> p3;
      p3 = Point<image_region_dimensions>::ONES();
      Point <image_region_dimensions> p4;
      p4 = p1 - p3;
      Rect<image_region_dimensions> r(p0, p2);
      Rect<image_region_dimensions> imageBounds(mImageDescriptor.origin(), mImageDescriptor.upperBound() - Point<image_region_dimensions>(1));
      domain = Domain(imageBounds);
      indexSpace = mRuntime->create_index_space(context, domain);
      fields = imageFields(context);
      region = mRuntime->create_logical_region(context, indexSpace, fields);
      fieldID[0] = FID_FIELD_R;
      fieldID[1] = FID_FIELD_G;
      fieldID[2] = FID_FIELD_B;
      fieldID[3] = FID_FIELD_A;
      fieldID[4] = FID_FIELD_Z;
      fieldID[5] = FID_FIELD_USERDATA;
    }
    
#if 0
    void ImageReduction::partitionImageByDepth(LogicalRegion image, Domain &domain, LogicalPartition &partition, Context context) {
      IndexSpaceT<image_region_dimensions> parent(image.get_index_space());
      Point<image_region_dimensions> blockingFactor = mImageDescriptor.layerSize();
      IndexPartition imageDepthIndexPartition = mRuntime->create_partition_by_blockify(context, parent, blockingFactor);
      mDepthPartitionColorSpace =
        CObjectWrapper::unwrap(legion_index_partition_get_color_space(CObjectWrapper::wrap(mRuntime), CObjectWrapper::wrap(imageDepthIndexPartition)));
      partition = mRuntime->get_logical_partition(context, image, imageDepthIndexPartition);
      mRuntime->attach_name(partition, "image depth partition");
      Rect<image_region_dimensions> depthBounds(mImageDescriptor.origin(), mImageDescriptor.numLayers() - Point<image_region_dimensions>(1));
      domain = Domain(depthBounds);
    }
#endif
    
    
    void ImageReduction::partitionImageEverywhere(LogicalRegion image, Domain& domain, LogicalPartition& partition, Context ctx, HighLevelRuntime* runtime, ImageDescriptor imageDescriptor) {
      
      int nodeCount = runtime->get_tunable_value(ctx, DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT, mMapperID);
      int cpuCount = runtime->get_tunable_value(ctx, DefaultMapper::DEFAULT_TUNABLE_GLOBAL_CPUS, mMapperID);
      int cpusPerNode = cpuCount / nodeCount;
      
      Point<image_region_dimensions> p0;
      p0 = mImageDescriptor.origin();
      Point <image_region_dimensions> p1;
      p1[0] = 0;
      p1[1] = 0;
      p1[2] = mImageDescriptor.numImageLayers - 1;
#if 1
      std::cout << __FUNCTION__ << " p0 " << p0 << " p1 " << p1 << std::endl;
#endif
      Rect<image_region_dimensions> color_bounds(p0, p1);
      IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);
      IndexSpace is = image.get_index_space();
      IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
      runtime->attach_name(ip, "ip");
      partition = runtime->get_logical_partition(ctx, image, ip);
      mRuntime->attach_name(partition, "image everywhere partition");
      p1 = mImageDescriptor.upperBound() - Point<image_region_dimensions>::ONES();
#if 1
      std::cout << __FUNCTION__ << "p0 " << p0 << " p1 " << p1 << std::endl;
#endif
      Rect<image_region_dimensions> everywhereBounds(p0, p1);
      domain = Domain(everywhereBounds);
#if 1
      std::cout << __FUNCTION__ << "p0 " << p0 << " p1 " << p1 << std::endl;
      std::cout << __FUNCTION__ << " bounds " << everywhereBounds << std::endl;
#endif
    }
    
#if 0
    void ImageReduction::partitionImageByFragment(LogicalRegion image, Domain &domain, LogicalPartition &partition, Context context) {
      IndexSpaceT<image_region_dimensions> parent(image.get_index_space());
      Point<image_region_dimensions> fragmentSize = { mImageDescriptor.width, mImageDescriptor.height, 1 };
      Point<image_region_dimensions> blockingFactor = fragmentSize;
      IndexPartition imageFragmentIndexPartition = mRuntime->create_partition_by_blockify(context, parent, blockingFactor);
      mRuntime->attach_name(imageFragmentIndexPartition, "image fragment index");
      partition = mRuntime->get_logical_partition(context, image, imageFragmentIndexPartition);
      mRuntime->attach_name(partition, "image fragment partition");
      Point<image_region_dimensions> numFragments = { 0, 0, mImageDescriptor.numImageLayers - 1 };
      Rect<image_region_dimensions> fragmentBounds(mImageDescriptor.origin(), numFragments);
      domain = Domain(fragmentBounds);
    }
#endif

    void ImageReduction::storeMyNodeID(int nodeID, int numNodes) {
      mNodeID = nodeID;
    }
    
    
    int ImageReduction::numTreeLevels(int numImageLayers) {
      int numTreeLevels = log2f(numImageLayers);
      if(powf(2.0f, numTreeLevels) < numImageLayers) {
        numTreeLevels++;
      }
      return numTreeLevels;
    }
    
    int ImageReduction::numTreeLevels(ImageDescriptor imageDescriptor) {
      return numTreeLevels(imageDescriptor.numImageLayers);
    }
    
    int ImageReduction::subtreeHeight(ImageDescriptor imageDescriptor) {
      const int totalLevels = numTreeLevels(imageDescriptor);
      const int MAX_LEVELS_PER_SUBTREE = 7; // 128 tasks per subtree
      return (totalLevels < MAX_LEVELS_PER_SUBTREE) ? totalLevels : MAX_LEVELS_PER_SUBTREE;
    }
    
    
    static int level2FunctorID(int level, int more) {
      return 100 + level * 2 + more;//TODO assign ids dynamically
    }
    
    
    void ImageReduction::createProjectionFunctors(int nodeID, Runtime* runtime, int numImageLayers) {
      
      // really need a lock here on mCompositeProjectionFunctor when running multithreaded locally
      // not a problem for multinode runs
      if(mCompositeProjectionFunctor == NULL) {
        mCompositeProjectionFunctor = new std::vector<CompositeProjectionFunctor*>();
        
        int numLevels = numTreeLevels(numImageLayers);
        int multiplier = numImageLayers;
        for(int level = 0; level < numLevels; ++level) {
          
          ProjectionID id0 = level2FunctorID(level, 0);
          int offset = 0;
          CompositeProjectionFunctor* functor0 = new CompositeProjectionFunctor(offset, multiplier, numImageLayers, id0);
          runtime->register_projection_functor(id0, functor0);
          mCompositeProjectionFunctor->push_back(functor0);
          
          ProjectionID id1 = level2FunctorID(level, 1);
          offset = multiplier / 2;
          CompositeProjectionFunctor* functor1 = new CompositeProjectionFunctor(offset, multiplier, numImageLayers, id1);
          runtime->register_projection_functor(id1, functor1);
          mCompositeProjectionFunctor->push_back(functor1);
          
          multiplier /= 2;
        }
      }
    }
    
    
    
    
    void ImageReduction::initial_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, HighLevelRuntime *runtime) {
      
#ifdef TRACE_TASKS
      std::cout << describe_task(task) << std::endl;
#endif
      
      Processor processor = runtime->get_executing_processor(ctx);
      Machine::ProcessorQuery query(Machine::get_machine());
      query.only_kind(processor.kind());
      {
        // set the node ID
        Domain indexSpaceDomain = runtime->get_index_space_domain(regions[0].get_logical_region().get_index_space());
        Rect<image_region_dimensions> imageBounds = indexSpaceDomain;
        int myNodeID = imageBounds.lo[2];        
        ImageDescriptor imageDescriptor = ((ImageDescriptor*)task->args)[0];
        storeMyNodeID(myNodeID, imageDescriptor.numImageLayers);
        createProjectionFunctors(myNodeID, runtime, imageDescriptor.numImageLayers);
      }
    }
    
    
    void ImageReduction::initializeNodes(HighLevelRuntime* runtime, Context context) {
      launch_task_everywhere(mInitialTaskID, runtime, context, NULL, 0, true);
    }
    
    
    void ImageReduction::initializeViewMatrix() {
      memset(mGlViewTransform, 0, sizeof(mGlViewTransform));
      mGlViewTransform[0] = mGlViewTransform[5] = mGlViewTransform[10] = mGlViewTransform[15] = 1.0f;
    }
    
    
    void ImageReduction::createTreeDomains(int nodeID, int numTreeLevels, Runtime* runtime, ImageDescriptor imageDescriptor) {
      if(mHierarchicalTreeDomain == NULL) {
        mHierarchicalTreeDomain = new std::vector<Domain>();
      }
      
      std::cout << "image descriptor " << mImageDescriptor.width << " " << mImageDescriptor.height << " " << mImageDescriptor.numImageLayers << std::endl;
      Point<image_region_dimensions> numFragments = { 0, 0, mImageDescriptor.numImageLayers - 1 };
      int numLeaves = 1;
      
      for(int level = 0; level < numTreeLevels; ++level) {
        if((unsigned)level >= mHierarchicalTreeDomain->size()) {
          numFragments[2] = numLeaves - 1;
          Rect<image_region_dimensions> launchBounds(Point<image_region_dimensions>::ZEROES(), numFragments);
          Domain domain = Domain(launchBounds);
std::cout << __FUNCTION__ << " tree level " << level << " domain " << domain << std::endl;
          mHierarchicalTreeDomain->push_back(domain);
        }
        numLeaves *= 2;
      }
      
    }
    
    
    void ImageReduction::addImageFieldsToRequirement(RegionRequirement &req) {
      req.add_field(FID_FIELD_R);
      req.add_field(FID_FIELD_G);
      req.add_field(FID_FIELD_B);
      req.add_field(FID_FIELD_A);
      req.add_field(FID_FIELD_Z);
      req.add_field(FID_FIELD_USERDATA);
    }
    
    
    
    void ImageReduction::create_image_field_pointers(ImageDescriptor imageDescriptor,
                                                     PhysicalRegion region,
                                                     PixelField *&r,
                                                     PixelField *&g,
                                                     PixelField *&b,
                                                     PixelField *&a,
                                                     PixelField *&z,
                                                     PixelField *&userdata,
                                                     Stride stride,
                                                     Runtime *runtime,
                                                     Context context,
                                                     bool readWrite) {
      
      const Rect<image_region_dimensions> rect = runtime->get_index_space_domain(context,
                                                                                 region.get_logical_region().get_index_space());
      
      if(readWrite) {
        const FieldAccessor<READ_WRITE, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_r(region, FID_FIELD_R);
        const FieldAccessor<READ_WRITE, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_g(region, FID_FIELD_G);
        const FieldAccessor<READ_WRITE, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_b(region, FID_FIELD_B);
        const FieldAccessor<READ_WRITE, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_a(region, FID_FIELD_A);
        const FieldAccessor<READ_WRITE, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_z(region, FID_FIELD_Z);
        const FieldAccessor<READ_WRITE, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_userdata(region, FID_FIELD_USERDATA);
        r = acc_r.ptr(rect, stride[0]);
        g = acc_g.ptr(rect, stride[1]);
        b = acc_b.ptr(rect, stride[2]);
        a = acc_a.ptr(rect, stride[3]);
        z = acc_z.ptr(rect, stride[4]);
        userdata = acc_userdata.ptr(rect, stride[5]);
        
      } else {
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_r(region, FID_FIELD_R);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_g(region, FID_FIELD_G);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_b(region, FID_FIELD_B);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_a(region, FID_FIELD_A);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_z(region, FID_FIELD_Z);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_userdata(region, FID_FIELD_USERDATA);
        r = (PixelField*)acc_r.ptr(rect, stride[0]);
        g = (PixelField*)acc_g.ptr(rect, stride[1]);
        b = (PixelField*)acc_b.ptr(rect, stride[2]);
        a = (PixelField*)acc_a.ptr(rect, stride[3]);
        z = (PixelField*)acc_z.ptr(rect, stride[4]);
        userdata = (PixelField*)acc_userdata.ptr(rect, stride[5]);
        
      }
      
    }
    
#if 0
    FutureMap ImageReduction::launch_index_task_by_depth(unsigned taskID, HighLevelRuntime* runtime, Context context, void *args, int argLen, bool blocking){
      
      ArgumentMap argMap;
      int totalArgLen = sizeof(mImageDescriptor) + argLen;
      char *argsBuffer = new char[totalArgLen];
      memcpy(argsBuffer, &mImageDescriptor, sizeof(mImageDescriptor));
      if(argLen > 0) {
        memcpy(argsBuffer + sizeof(mImageDescriptor), args, argLen);
      }
      
      IndexTaskLauncher depthLauncher(taskID, mDepthDomain, TaskArgument(argsBuffer, totalArgLen), argMap, Predicate::TRUE_PRED, false, mMapperID);
      RegionRequirement req(mDepthPartition, 0, READ_WRITE, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      depthLauncher.add_region_requirement(req);
      FutureMap futures = mRuntime->execute_index_space(context, depthLauncher);
      
      if(blocking) {
        futures.wait_all_results();
      }
      delete [] argsBuffer;
      return futures;
    }
#endif
    
    
    FutureMap ImageReduction::launch_task_everywhere(unsigned taskID, HighLevelRuntime* runtime, Context context, void *args, int argLen, bool blocking){
      
      ArgumentMap argMap;
      
      __TRACE
      
      int totalArgLen = sizeof(mImageDescriptor) + argLen;
      char *argsBuffer = new char[totalArgLen];
      memcpy(argsBuffer, &mImageDescriptor, sizeof(mImageDescriptor));
      if(argLen > 0) {
        memcpy(argsBuffer + sizeof(mImageDescriptor), args, argLen);
      }
      
      IndexTaskLauncher everywhereLauncher(taskID, mEverywhereDomain, TaskArgument(argsBuffer, totalArgLen), argMap, Predicate::TRUE_PRED, false, mMapperID);
      RegionRequirement req(mEverywherePartition, 0, READ_WRITE, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      everywhereLauncher.add_region_requirement(req);
      
      __TRACE
      FutureMap futures = runtime->execute_index_space(context, everywhereLauncher);
      
      if(blocking) {
        futures.wait_all_results();
      }
      delete [] argsBuffer;
      return futures;
    }
    
    
#ifdef DEBUG
    static void dumpImage(ImageReduction::PixelField *rr, ImageReduction::PixelField*gg, ImageReduction::PixelField*bb, ImageReduction::PixelField*aa, ImageReduction::PixelField*zz, ImageReduction::PixelField*uu, ImageReduction::Stride stride, char *text) {
      std::cout << std::endl;
      std::cout << text << std::endl;
      for(int i = 0; i < 10; ++i) {
        std::cout << text << " pixel " << i << ": ";
        std::cout << rr[0] << "\t" << gg[0] << "\t" << bb[0] << "\t" << aa[0] << "\t" << zz[0] << "\t" << uu[0] << std::endl;
        ImageReductionComposite::increment(rr, gg, bb, aa, zz, uu, stride);
      }
    }
#endif
    
    
    static bool loop = true;
    
    void ImageReduction::composite_task(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context ctx, HighLevelRuntime *runtime) {
#ifdef TRACE_TASKS
      std::cout << describe_task(task) << std::endl;
#endif
      
#if NULL_COMPOSITE_TASKS
      return;//performance testing
#endif
      
      CompositeArguments args = ((CompositeArguments*)task->args)[0];
      PhysicalRegion fragment0 = regions[0];
      PhysicalRegion fragment1 = regions[1];
      
      Stride stride0, stride1;
      PixelField *r0, *g0, *b0, *a0, *z0, *userdata0;
      PixelField *r1, *g1, *b1, *a1, *z1, *userdata1;
      ImageReductionComposite::CompositeFunction* compositeFunction;
      //      UsecTimer composite("composite time:");
      //      composite.start();
      create_image_field_pointers(args.imageDescriptor, fragment0, r0, g0, b0, a0, z0, userdata0, stride0, runtime, ctx, true);
      create_image_field_pointers(args.imageDescriptor, fragment1, r1, g1, b1, a1, z1, userdata1, stride1, runtime, ctx, false);
      
#if 1
        PixelField* rr = r0;
        PixelField* gg = g0;
        PixelField* bb = b0;
        PixelField* aa = a0;
        PixelField* zz = z0;
        PixelField* uu = userdata0;
#endif
      
#if 1
      PixelField* rr0 = r0;
      PixelField* gg0 = g0;
      PixelField* bb0 = b0;
      PixelField* aa0 = a0;
      PixelField* zz0 = z0;
      PixelField* uu0 = userdata0;
      std::cout << "composite image 0" << std::endl;
      for(unsigned i = 0; i < 64; ++i) {
        std::cout << "pixel " << i << " : " << *rr0 << " " << *gg0 << " " << *bb0 << " " << *aa0 << " " << *zz0 << " " << *uu0 << std::endl;
        ImageReductionComposite::increment(rr0, gg0, bb0, aa0, zz0, uu0, stride0);
      }
      PixelField* rr1 = r1;
      PixelField* gg1 = g1;
      PixelField* bb1 = b1;
      PixelField* aa1 = a1;
      PixelField* zz1 = z1;
      PixelField* uu1 = userdata1;
      std::cout << "composite image 1" << std::endl;
      for(unsigned i = 0; i < 64; ++i) {
        std::cout << "pixel " << i << " : " << *rr1 << " " << *gg1 << " " << *bb1 << " " << *aa1 << " " << *zz1 << " " << *uu1 << std::endl;
        ImageReductionComposite::increment(rr1, gg1, bb1, aa1, zz1, uu1, stride1);
      }
#endif
      
#if 0
      std::cout << "composite_task pointer0 " << r0 << " " << g0 << " " << b0 << " " << a0 << " " << z0 << " " << userdata0 << std::endl;
      std::cout << "composite_task pointer1 " << r1 << " " << g1 << " " << b1 << " " << a1 << " " << z1 << " " << userdata1 << std::endl;
      PixelField* delta = (PixelField*)0x1800;
      PixelField* newBase = (PixelField*)((long long int)r0 - (long long int)delta);
      PixelField* newBase96 = (PixelField*)((long long int)newBase + 96);
      std::cout << "base of result region should be at " << newBase << " also " << newBase96 << std::endl;
      std::cout << "watch *(float*)" << newBase << std::endl;
      std::cout << "watch *(float*)" << newBase96 << std::endl;
#if 0
      PixelField* rr = newBase;
      std::cout << "here is the data already in the result buffer" << std::endl;
      for(unsigned i = 0; i < 64; ++i) {
        std::cout << "pixel " << i << " : " << *rr << std::endl;
        rr = rr + stride0[0][0];
      }
#endif
      if(loop) std::cout << "looping here for debugger" << std::endl;
      while(loop) {
        if(!loop)
          break;
      }
      std::cout << "exiting debug loop" << std::endl;
#endif

      compositeFunction = ImageReductionComposite::compositeFunctionPointer(args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination, args.blendEquation);
      compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0, args.imageDescriptor.pixelsPerLayer(), stride0, stride1);
      //      composite.stop();
      //      std::cout << composite.to_string() << std::endl;

#if 1
      std::cout << "after composite_task with pointer0 " << rr << " " << gg << " " << bb << " " << aa << " " << zz << "" << uu << std::endl;
      for(unsigned i = 0; i < 64; ++i) {
        std::cout << "pixel " << i << " : " << *rr << " " << *gg << " " << *bb << " " << *aa << " " << *zz << " " << *uu << std::endl;
        ImageReductionComposite::increment(rr, gg, bb, aa, zz, uu, stride0);
      }
#endif
    }
    
    
    
    
    
    FutureMap ImageReduction::launchTreeReduction(ImageDescriptor imageDescriptor, int treeLevel,
                                                  GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation,
                                                  int compositeTaskID, LogicalPartition sourcePartition, LogicalRegion image,
                                                  Runtime* runtime, Context context,
                                                  int nodeID, int maxTreeLevel) {
      Domain launchDomain = (*mHierarchicalTreeDomain)[treeLevel - 1];
      int index = (treeLevel - 1) * 2;
      CompositeProjectionFunctor* functor0 = (*mCompositeProjectionFunctor)[index];
      CompositeProjectionFunctor* functor1 = (*mCompositeProjectionFunctor)[index + 1];
      
#if 1
      std::cout << __FUNCTION__ << " tree level " << treeLevel << " using functors " << functor0->to_string() << " " << functor1->to_string() << std::endl;
      std::cout << __FUNCTION__ << " launch domain at tree level " << treeLevel
      << " " << launchDomain << std::endl;
#endif
      
      ArgumentMap argMap;
      CompositeArguments args = { imageDescriptor, depthFunc, blendFuncSource, blendFuncDestination, blendEquation };
      IndexTaskLauncher treeCompositeLauncher(compositeTaskID, launchDomain, TaskArgument(&args, sizeof(args)), argMap, Predicate::TRUE_PRED, false, gMapperID);
      
      RegionRequirement req0(sourcePartition, functor0->id(), READ_WRITE, EXCLUSIVE, image);
      addImageFieldsToRequirement(req0);
      treeCompositeLauncher.add_region_requirement(req0);
      
      RegionRequirement req1(sourcePartition, functor1->id(), READ_ONLY, EXCLUSIVE, image);
      addImageFieldsToRequirement(req1);
      treeCompositeLauncher.add_region_requirement(req1);
      
      FutureMap futures = runtime->execute_index_space(context, treeCompositeLauncher);
      
      if(treeLevel > 1) {
        
        futures = launchTreeReduction(imageDescriptor, treeLevel - 1, depthFunc, blendFuncSource, blendFuncDestination, blendEquation, compositeTaskID, sourcePartition, image, runtime, context, nodeID, maxTreeLevel);
      }
      
      return futures;
      
    }
    
    
    
    FutureMap ImageReduction::reduceAssociative(Context context) {
      int maxTreeLevel = numTreeLevels(mImageDescriptor);
      if(maxTreeLevel > 0) {
        return launchTreeReduction(mImageDescriptor, maxTreeLevel, mDepthFunction, mGlBlendFunctionSource, mGlBlendFunctionDestination, mGlBlendEquation,
                                   mCompositeTaskID, mEverywherePartition, mSourceImage,
                                   mRuntime, context, mNodeID, maxTreeLevel);
      } else {
        return FutureMap();
      }
    }
    
    
    FutureMap ImageReduction::reduce_associative_commutative(Context context){
      return reduceAssociative(context);
    }
    
    FutureMap ImageReduction::reduce_associative_noncommutative(Context context){
      if(mNumSimulationBounds == mImageDescriptor.numImageLayers) {
        return reduceAssociative(context);
      } else {
        std::cout << "cannot reduce noncommutatively until simulation bounds are provided" << std::endl;
        std::cout << "call ImageReduction::preregisterSimulationBounds(SimulationBoundsCoordinate *bounds, int numBounds) before starting Legion runtime" << std::endl;
        return FutureMap();
      }
    }
    
    
    
    FutureMap ImageReduction::launchPipelineReduction() {
      return FutureMap();
    }
    
    
    
    FutureMap ImageReduction::reduceNonassociative(Context context) {
      // NOP - TODO
      return launchPipelineReduction();
    }
    
    FutureMap ImageReduction::reduce_nonassociative_commutative(Context context){
      return reduceNonassociative(context);
    }
    
    FutureMap ImageReduction::reduce_nonassociative_noncommutative(Context context){
      return reduceNonassociative(context);
    }
    
    //project pixels from eye space bounded by l,r,t,p,n,f to clip space -1,1
    
    ImageReduction::SimulationBoundsCoordinate *homogeneousOrthographicProjectionMatrix(ImageReduction::SimulationBoundsCoordinate right,
                                                                                        ImageReduction::SimulationBoundsCoordinate left,
                                                                                        ImageReduction::SimulationBoundsCoordinate top,
                                                                                        ImageReduction::SimulationBoundsCoordinate bottom,
                                                                                        ImageReduction::SimulationBoundsCoordinate far,
                                                                                        ImageReduction::SimulationBoundsCoordinate near) {
      static ImageReduction::SimulationBoundsCoordinate *result = NULL;
      if(result == NULL) {
        result = new ImageReduction::SimulationBoundsCoordinate[ImageReduction::numMatrixElements4x4];
        // row major
        result[0] = (2.0f / (right - left));
        result[1] = 0.0f;
        result[2] = 0.0f;
        result[3] = 0.0f;
        //
        result[4] = 0.0f;
        result[5] = (2.0f / (top - bottom));
        result[6] = 0.0f;
        result[7] = 0.0f;
        //
        result[8] = 0.0f;
        result[9] = 0.0f;
        result[10] = (-2.0f / (far - near));
        result[11] = 0.0f;
        //
        result[12] = -((right + left) / (right - left));
        result[13] = -((top + bottom) / (top - bottom));
        result[14] = -((far + near) / (far - near));
        result[15] = 1.0f;
      }
      return result;
    }
    
    
    // perspective projection
    
    ImageReduction::SimulationBoundsCoordinate *homogeneousPerspectiveProjectionMatrix(                                                                                              ImageReduction::SimulationBoundsCoordinate left,
                                                                                       ImageReduction::SimulationBoundsCoordinate right,
                                                                                       ImageReduction::SimulationBoundsCoordinate bottom,
                                                                                       ImageReduction::SimulationBoundsCoordinate top,
                                                                                       ImageReduction::SimulationBoundsCoordinate zNear,
                                                                                       ImageReduction::SimulationBoundsCoordinate zFar)
    {
      static ImageReduction::SimulationBoundsCoordinate *result = NULL;
      if(result == NULL) {
        result = new ImageReduction::SimulationBoundsCoordinate[ImageReduction::numMatrixElements4x4];
        // row major
        result[0] = 2.0f * zNear / (right - left);
        result[1] = 0.0f;
        ImageReduction::SimulationBoundsCoordinate A = (right + left) / (right - left);
        result[2] = A;
        result[3] = 0.0f;
        //
        result[4] = 0.0f;
        result[5] = 2.0f * zNear / (top - bottom);
        ImageReduction::SimulationBoundsCoordinate B = (top + bottom) / (top - bottom);
        result[6] = B;
        result[7] = 0.0f;
        //
        result[8] = 0.0f;
        result[9] = 0.0f;
        ImageReduction::SimulationBoundsCoordinate C = -(zFar + zNear) / (zFar - zNear);
        result[10] = C;
        ImageReduction::SimulationBoundsCoordinate D = -(2.0f * zFar * zNear) / (zFar - zNear);
        result[11] = D;
        //
        result[12] = 0.0f;
        result[13] = 0.0f;
        result[14] = -1.0f;
        result[15] = 0.0f;
      }
      return result;
    }
    
    
    static void normalize3(ImageReduction::SimulationBoundsCoordinate x[3],
                           ImageReduction::SimulationBoundsCoordinate y[3]) {
      ImageReduction::SimulationBoundsCoordinate norm = sqrtf(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      y[0] = x[0] / norm;
      y[1] = x[1] / norm;
      y[2] = x[2] / norm;
    }
    
    
    // s = u X v
    static void cross3(ImageReduction::SimulationBoundsCoordinate u[3],
                       ImageReduction::SimulationBoundsCoordinate v[3],
                       ImageReduction::SimulationBoundsCoordinate s[3]) {
      s[0] = u[1] * v[2] - u[2] * v[1];
      s[1] = u[2] * v[0] - u[0] * v[2];
      s[2] = u[0] * v[1] - u[1] * v[0];
    }
    
    // based on gluLookAt
    ImageReduction::SimulationBoundsCoordinate *viewTransform(
                                                              ImageReduction::SimulationBoundsCoordinate eyeX,
                                                              ImageReduction::SimulationBoundsCoordinate eyeY,
                                                              ImageReduction::SimulationBoundsCoordinate eyeZ,
                                                              ImageReduction::SimulationBoundsCoordinate centerX,
                                                              ImageReduction::SimulationBoundsCoordinate centerY,
                                                              ImageReduction::SimulationBoundsCoordinate centerZ,
                                                              ImageReduction::SimulationBoundsCoordinate upX,
                                                              ImageReduction::SimulationBoundsCoordinate upY,
                                                              ImageReduction::SimulationBoundsCoordinate upZ)
    {
      static ImageReduction::SimulationBoundsCoordinate *result = NULL;
      if(result == NULL) {
        result = new ImageReduction::SimulationBoundsCoordinate[ImageReduction::numMatrixElements4x4];
        
        ImageReduction::SimulationBoundsCoordinate F[3] = {
          centerX - eyeX, centerY - eyeY, centerZ - eyeZ
        };
        ImageReduction::SimulationBoundsCoordinate UP[3] = {
          upX, upY, upZ
        };
        ImageReduction::SimulationBoundsCoordinate f[3];
        normalize3(F, f);
        ImageReduction::SimulationBoundsCoordinate UPprime[3];
        normalize3(UP, UPprime);
        
        ImageReduction::SimulationBoundsCoordinate s[3];
        cross3(f, UPprime, s);
        
        ImageReduction::SimulationBoundsCoordinate u[3];
        cross3(s, f, u);
        
        // row major
        result[0] = s[0];
        result[1] = s[1];
        result[2] = s[2];
        result[3] = 0.0f;
        //
        result[4] = u[0];
        result[5] = u[1];
        result[6] = u[2];
        result[7] = 0.0f;
        //
        result[8] = -f[0];
        result[9] = -f[1];
        result[10] = -f[2];
        result[11] = 0.0f;
        //
        result[12] = 0.0f;
        result[13] = 0.0f;
        result[14] = 0.0f;
        result[15] = 1.0f;
      }
      return result;
    }
    
    
#if 0 // save until needed
    static void matrixMultiply4x4(ImageReduction::SimulationBoundsCoordinate* A,
                                  ImageReduction::SimulationBoundsCoordinate* B,
                                  ImageReduction::SimulationBoundsCoordinate* C) {
      // C = A x B
      // https://stackoverflow.com/questions/18499971/efficient-4x4-matrix-multiplication-c-vs-assembly
      for (unsigned int i = 0; i < 16; i += 4)
        for (unsigned int j = 0; j < 4; ++j)
          C[i + j] = (B[i + 0] * A[j +  0])
          + (B[i + 1] * A[j +  4])
          + (B[i + 2] * A[j +  8])
          + (B[i + 3] * A[j + 12]);
    }
#endif
    
    
    
    
    void ImageReduction::display_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, HighLevelRuntime *runtime) {
      
#ifdef TRACE_TASKS
      std::cout << describe_task(task) << std::endl;
#endif
      DisplayArguments args = ((DisplayArguments*)task->args)[0];
      char fileName[1024];
      sprintf(fileName, "display.%d.txt", args.t);
      string outputFileName = string(fileName);
      UsecTimer display(describe_task(task) + " write to " + outputFileName + ":");
      display.start();
      PhysicalRegion displayPlane = regions[0];
      Stride stride;
      PixelField *r, *g, *b, *a, *z, *userdata;
      create_image_field_pointers(args.imageDescriptor, displayPlane, r, g, b, a, z, userdata, stride, runtime, ctx, false);
      
      FILE *outputFile = fopen(outputFileName.c_str(), "wb");
      fwrite(r, numPixelFields * sizeof(*r), args.imageDescriptor.pixelsPerLayer(), outputFile);
      fclose(outputFile);
      
      display.stop();
      cout << display.to_string() << endl;
    }
    
    
    
    Future ImageReduction::display(int t, Context context) {
      DisplayArguments args = { mImageDescriptor, t };
      TaskLauncher taskLauncher(mDisplayTaskID, TaskArgument(&args, sizeof(args)));
      DomainPoint origin = DomainPoint(Point<image_region_dimensions>::ZEROES());
      LogicalRegion displayPlane = mRuntime->get_logical_subregion_by_color(mEverywherePartition, origin);
      RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      taskLauncher.add_region_requirement(req);
      Future displayFuture = mRuntime->execute_task(context, taskLauncher);
      return displayFuture;
    }
    
  }
}
