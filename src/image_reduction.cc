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

#include "image_reduction.h"
#include "image_reduction_composite.h"

#include <iostream>
#include <fstream>
#include <math.h>


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


namespace Legion {
  namespace Visualization {
    
    
    // declare module static data
    
    int *ImageReduction::mNodeID;
    ImageReduction::SimulationBoundsCoordinate *ImageReduction::mSimulationBounds;
    int ImageReduction::mNumSimulationBounds;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mXMax;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mXMin;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mYMax;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mYMin;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mZMax;
    ImageReduction::SimulationBoundsCoordinate ImageReduction::mZMin;
    std::vector<ImageReduction::CompositeProjectionFunctor*> *ImageReduction::mCompositeProjectionFunctor = NULL;
    int ImageReduction::mNodeCount;
    std::vector<Domain> *ImageReduction::mHierarchicalTreeDomain = NULL;
    GLfloat ImageReduction::mGlViewTransform[numMatrixElements4x4];
    ImageReduction::PixelField ImageReduction::mGlConstantColor[numPixelFields];
    GLenum ImageReduction::mGlBlendEquation;
    GLenum ImageReduction::mGlBlendFunctionSource;
    GLenum ImageReduction::mGlBlendFunctionDestination;
    TaskID ImageReduction::mInitialTaskID;
    TaskID ImageReduction::mCompositeTaskID;
    TaskID ImageReduction::mDisplayTaskID;
    
    
    ImageReduction::ImageReduction(ImageSize imageSize, Context context, HighLevelRuntime *runtime) {
      mImageSize = imageSize;
      mContext = context;
      mRuntime = runtime;
      mDepthFunction = 0;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mNodeID = NULL;
      
      mGlBlendEquation = GL_FUNC_ADD;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mDepthFunction = 0;
      
      createImage(mSourceImage, mSourceImageDomain);
      partitionImageByDepth(mSourceImage, mDepthDomain, mDepthPartition);
      partitionImageByFragment(mSourceImage, mSourceFragmentDomain, mSourceFragmentPartition);
      
      initializeNodes(runtime, context);
      assert(mNodeCount > 0);
      mLocalCopyOfNodeID = mNodeID[mNodeCount - 1];//written by initial_task
      initializeViewMatrix();
      createTreeDomains(mLocalCopyOfNodeID, numTreeLevels(imageSize), runtime, imageSize);
      
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
      
      mRuntime->destroy_index_space(mContext, mSourceImage.get_index_space());
      mRuntime->destroy_logical_region(mContext, mSourceImage);
      mRuntime->destroy_index_partition(mContext, mDepthPartition.get_index_partition());
      mRuntime->destroy_logical_partition(mContext, mDepthPartition);
      mRuntime->destroy_index_partition(mContext, mSourceFragmentPartition.get_index_partition());
      mRuntime->destroy_logical_partition(mContext, mSourceFragmentPartition);
    }
    
    
    // this function should always be called prior to starting the Legion runtime

    void ImageReduction::initialize() {
      registerTasks();
    }
    
    
    // this function should be called prior to starting the Legion runtime if you
    // plan to use noncommutative reductions
    // its purpose is to copy the domain bounds to all nodes
    
    void ImageReduction::preregisterSimulationBounds(SimulationBoundsCoordinate *bounds, int numBounds) {
      mNodeCount = 0;
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
      std::cout << "loaded " << numBounds << " simulation subdomains, overall bounds ("
      << mXMin << "," << mXMax << " x " << mYMin << "," << mYMax << " x " << mZMin << "," << mZMax << ")" << std::endl;
      
    }
    
    // this function should be called prior to starting the Legion runtime
    // its purpose is to register tasks with the same id on all nodes
    
    void ImageReduction::registerTasks() {
      
      mInitialTaskID = Legion::HighLevelRuntime::generate_static_task_id();
      Legion::HighLevelRuntime::register_legion_task<initial_task>(mInitialTaskID,
                                                                   Legion::Processor::LOC_PROC, false/*single*/, true/*index*/,                                                                                                  AUTO_GENERATE_ID, Legion::TaskConfigOptions(true/*leaf*/), "initial_task");
      
      mCompositeTaskID = Legion::HighLevelRuntime::generate_static_task_id();
      Legion::HighLevelRuntime::register_legion_task<composite_task>(mCompositeTaskID,
                                                                     Legion::Processor::LOC_PROC, false/*single*/, true/*index*/,                                                                                                  AUTO_GENERATE_ID, Legion::TaskConfigOptions(true/*leaf*/), "composite_task");
      
      mDisplayTaskID = Legion::HighLevelRuntime::generate_static_task_id();
      Legion::HighLevelRuntime::register_legion_task<display_task>(mDisplayTaskID,
                                                                   Legion::Processor::LOC_PROC, true/*single*/, false/*index*/,                                                                                                  AUTO_GENERATE_ID, Legion::TaskConfigOptions(true/*leaf*/), "display_task");
    }
    
    
    FieldSpace ImageReduction::imageFields() {
      FieldSpace fields = mRuntime->create_field_space(mContext);
      mRuntime->attach_name(fields, "pixel fields");
      {
        FieldAllocator allocator = mRuntime->create_field_allocator(mContext, fields);
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
    
    
    void ImageReduction::createImage(LogicalRegion &region, Domain &domain) {
      Point<image_region_dimensions> p0;
      p0 = mImageSize.origin();
      Point <image_region_dimensions> p1;
      p1 = mImageSize.upperBound();
      Point <image_region_dimensions> p2(1);
      p2 = Point<image_region_dimensions>(1);
      p2 = mImageSize.upperBound() - Point<image_region_dimensions>(1);
      Point <image_region_dimensions> p3;
      p3 = Point<image_region_dimensions>::ONES();
      Point <image_region_dimensions> p4;
      p4 = p1 - p3;
      Rect<image_region_dimensions> r(p0, p2);
      Rect<image_region_dimensions> imageBounds(mImageSize.origin(), mImageSize.upperBound() - Point<image_region_dimensions>(1));
      domain = Domain(imageBounds);
      IndexSpace pixels = mRuntime->create_index_space(mContext, domain);
      FieldSpace fields = imageFields();
      region = mRuntime->create_logical_region(mContext, pixels, fields);
    }
    
    
    void ImageReduction::partitionImageByDepth(LogicalRegion image, Domain &domain, LogicalPartition &partition) {
      IndexSpaceT<image_region_dimensions> parent(image.get_index_space());
      Point<image_region_dimensions> blockingFactor = mImageSize.layerSize();
      IndexPartition imageDepthIndexPartition = mRuntime->create_partition_by_blockify(mContext, parent, blockingFactor);
      partition = mRuntime->get_logical_partition(mContext, image, imageDepthIndexPartition);
      mRuntime->attach_name(partition, "image depth partition");
      Rect<image_region_dimensions> depthBounds(mImageSize.origin(), mImageSize.numLayers() - Point<image_region_dimensions>(1));
      domain = Domain(depthBounds);
    }
    
    
    void ImageReduction::partitionImageByFragment(LogicalRegion image, Domain &domain, LogicalPartition &partition) {
      IndexSpaceT<image_region_dimensions> parent(image.get_index_space());
      Point<image_region_dimensions> blockingFactor = mImageSize.fragmentSize();
      IndexPartition imageFragmentIndexPartition = mRuntime->create_partition_by_blockify(mContext, parent, blockingFactor);
      mRuntime->attach_name(imageFragmentIndexPartition, "image fragment index");
      partition = mRuntime->get_logical_partition(mContext, image, imageFragmentIndexPartition);
      mRuntime->attach_name(partition, "image fragment partition");
      Rect<image_region_dimensions> fragmentBounds(mImageSize.origin(), mImageSize.numFragments() - Point<image_region_dimensions>(1));
      domain = Domain(fragmentBounds);
    }
    
    
    ///////////////
    //FIXME awkwardness about running multithreaded versus multinode can this be removed
    
    void ImageReduction::storeMyNodeID(int nodeID, int numNodes) {
      if(mNodeID == NULL) {
        mNodeID = new int[numNodes];
      }
      mNodeID[nodeID] = nodeID;
      mNodeCount++;
    }
    
    ////////////////
    
    
    
    int ImageReduction::numTreeLevels(int numImageLayers) {
      int numTreeLevels = log2f(numImageLayers);
      if(powf(2.0f, numTreeLevels) < numImageLayers) {
        numTreeLevels++;
      }
      return numTreeLevels;
    }
    
    int ImageReduction::numTreeLevels(ImageSize imageSize) {
      return numTreeLevels(imageSize.numImageLayers);
    }

    int ImageReduction::subtreeHeight(ImageSize imageSize) {
      const int totalLevels = numTreeLevels(imageSize);
      const int MAX_LEVELS_PER_SUBTREE = 7; // 128 tasks per subtree
      return (totalLevels < MAX_LEVELS_PER_SUBTREE) ? totalLevels : MAX_LEVELS_PER_SUBTREE;
    }
    
    
    static int level2FunctorID(int level, int more) {
      return 100 + level * 2 + more;//TODO assign ids dynamically
    }
    
    
    void ImageReduction::createProjectionFunctors(int nodeID, Runtime* runtime, int numImageLayers) {
      
      std::cout << __FUNCTION__ << std::endl;
      
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
          
          multiplier /= num_fragments_per_composite;
        }
      }
    }
    
    
    
    // this task and everything it calls is invoked on every node during initialization
    
    void ImageReduction::initial_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, HighLevelRuntime *runtime) {
      
#ifdef TRACE_TASKS
      std::cout << describe_task(task) << std::endl;
#endif
      
      int *args = (int*)task->args;
      int numImageLayers = args[0];
      int myNodeID = args[1];
      storeMyNodeID(myNodeID, numImageLayers);

      Processor processor = runtime->get_executing_processor(ctx);
      Machine::ProcessorQuery query(Machine::get_machine());
      query.only_kind(processor.kind());
      std::cout << "query.first.id " << query.first().id << " processor.id " << processor.id << std::endl;
      if(query.first().id == processor.id) {
        // projection functors
        createProjectionFunctors(myNodeID, runtime, numImageLayers);
      }
    }
    
    
    void ImageReduction::initializeNodes(HighLevelRuntime* runtime, Context context) {
      launch_epoch_task_by_depth(mInitialTaskID, runtime, context, true);
    }
    
    
    void ImageReduction::initializeViewMatrix() {
      memset(mGlViewTransform, 0, sizeof(mGlViewTransform));
      mGlViewTransform[0] = mGlViewTransform[5] = mGlViewTransform[10] = mGlViewTransform[15] = 1.0f;
    }
    
    
    void ImageReduction::createTreeDomains(int nodeID, int numTreeLevels, Runtime* runtime, ImageSize imageSize) {
      if(mHierarchicalTreeDomain == NULL) {
        mHierarchicalTreeDomain = new std::vector<Domain>();
      }
      
      Point<image_region_dimensions> numFragments = imageSize.numFragments() - Point<image_region_dimensions>(1);
      int numLeaves = 1;
      
      for(int level = 0; level < numTreeLevels; ++level) {
        if((unsigned)level >= mHierarchicalTreeDomain->size()) {
          numFragments[2] = numLeaves - 1;
          Rect<image_region_dimensions> launchBounds(Point<image_region_dimensions>::ZEROES(), numFragments);
          Domain domain = Domain(launchBounds);
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
    
    
    void ImageReduction::createImageFieldPointer(RegionAccessor<AccessorType::Generic, PixelField> &acc,
                                                 int fieldID,
                                                 PixelField *&field,
                                                 Rect<image_region_dimensions> imageBounds,
                                                 PhysicalRegion region,
                                                 ByteOffset offset[image_region_dimensions]) {
      acc = region.get_field_accessor(fieldID).typeify<PixelField>();
      LegionRuntime::Arrays::Rect<image_region_dimensions> tempBounds;
      LegionRuntime::Arrays::Rect<image_region_dimensions> bounds = Domain(imageBounds).get_rect<image_region_dimensions>();
      field = acc.raw_rect_ptr<image_region_dimensions>(bounds, tempBounds, offset);
      assert(bounds == tempBounds);
    }
    
    
    void ImageReduction::create_image_field_pointers(ImageSize imageSize,
                                                     PhysicalRegion region,
                                                     PixelField *&r,
                                                     PixelField *&g,
                                                     PixelField *&b,
                                                     PixelField *&a,
                                                     PixelField *&z,
                                                     PixelField *&userdata,
                                                     Stride stride,
                                                     Runtime *runtime,
                                                     Context context) {
      
      Domain indexSpaceDomain = runtime->get_index_space_domain(context, region.get_logical_region().get_index_space());
      Rect<image_region_dimensions> imageBounds = indexSpaceDomain;
      
      RegionAccessor<AccessorType::Generic, PixelField> acc_r, acc_g, acc_b, acc_a, acc_z, acc_userdata;
      
      createImageFieldPointer(acc_r, FID_FIELD_R, r, imageBounds, region, stride[FID_FIELD_R]);
      createImageFieldPointer(acc_g, FID_FIELD_G, g, imageBounds, region, stride[FID_FIELD_G]);
      createImageFieldPointer(acc_b, FID_FIELD_B, b, imageBounds, region, stride[FID_FIELD_B]);
      createImageFieldPointer(acc_a, FID_FIELD_A, a, imageBounds, region, stride[FID_FIELD_A]);
      createImageFieldPointer(acc_z, FID_FIELD_Z, z, imageBounds, region, stride[FID_FIELD_Z]);
      createImageFieldPointer(acc_userdata, FID_FIELD_USERDATA, userdata, imageBounds, region, stride[FID_FIELD_USERDATA]);
    }
    
    
    FutureMap ImageReduction::launch_index_task_by_depth(unsigned taskID, HighLevelRuntime* runtime, Context context, void *args, int argLen, bool blocking){
      
      ArgumentMap argMap;
      int totalArgLen = sizeof(mImageSize) + argLen;
      char *argsBuffer = new char[totalArgLen];
      memcpy(argsBuffer, &mImageSize, sizeof(mImageSize));
      if(argLen > 0) {
        memcpy(argsBuffer + sizeof(mImageSize), args, argLen);
      }
      
      IndexTaskLauncher depthLauncher(taskID, mDepthDomain, TaskArgument(argsBuffer, totalArgLen), argMap);
      RegionRequirement req(mDepthPartition, 0, READ_WRITE, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      depthLauncher.add_region_requirement(req);
      FutureMap futures = mRuntime->execute_index_space(mContext, depthLauncher);

      if(blocking) {
        futures.wait_all_results();
      }
      delete [] argsBuffer;
      return futures;
    }
    
    FutureMap ImageReduction::launch_epoch_task_by_depth(unsigned taskID, HighLevelRuntime* runtime, Context context, bool blocking){
      
      MustEpochLauncher mustEpochLauncher;
      for(unsigned i = 0; i < mImageSize.numImageLayers; ++i) {
        int args[] = { mImageSize.numImageLayers, i };
        int argLen = 2 * sizeof(args[0]);
        TaskLauncher depthTaskLauncher(taskID, TaskArgument(args, argLen));
        DomainPoint point(i);
        mustEpochLauncher.add_single_task(point, depthTaskLauncher);
      }
      FutureMap futures = runtime->execute_must_epoch(context, mustEpochLauncher);
      
      if(blocking) {
        futures.wait_all_results();
      }
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
      
      
#if 0
      Domain indexSpaceDomain0 = runtime->get_index_space_domain(ctx, fragment0.get_logical_region().get_index_space());
      Rect<image_region_dimensions> imageBounds0 = indexSpaceDomain0.get_rect<image_region_dimensions>();
      Domain indexSpaceDomain1 = runtime->get_index_space_domain(ctx, fragment1.get_logical_region().get_index_space());
      Rect<image_region_dimensions> imageBounds1 = indexSpaceDomain1.get_rect<image_region_dimensions>();
      std::cout << describe_task(task) << " fragments " << imageBounds0 << " " << imageBounds1 << std::endl;
#endif
      
      
      Stride stride;
      PixelField *r0, *g0, *b0, *a0, *z0, *userdata0;
      PixelField *r1, *g1, *b1, *a1, *z1, *userdata1;
      ImageReductionComposite::CompositeFunction* compositeFunction;
      
      //      UsecTimer composite("composite time:");
      //      composite.start();
      create_image_field_pointers(args.imageSize, fragment0, r0, g0, b0, a0, z0, userdata0, stride, runtime, ctx);
      create_image_field_pointers(args.imageSize, fragment1, r1, g1, b1, a1, z1, userdata1, stride, runtime, ctx);
      compositeFunction = ImageReductionComposite::compositeFunctionPointer(args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination, args.blendEquation);
      compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0, args.imageSize.numPixelsPerFragment(), stride);
      //      composite.stop();
      //      std::cout << composite.to_string() << std::endl;
    }
    
    
    
    
    
    FutureMap ImageReduction::launchTreeReduction(ImageSize imageSize, int treeLevel,
                                                  GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation,
                                                  int compositeTaskID, LogicalPartition sourceFragmentPartition, LogicalRegion image,
                                                  Runtime* runtime, Context context,
                                                  int nodeID, int maxTreeLevel) {
      Domain launchDomain = (*mHierarchicalTreeDomain)[treeLevel - 1];
      int index = (treeLevel - 1) * 2;
      CompositeProjectionFunctor* functor0 = (*mCompositeProjectionFunctor)[index];
      CompositeProjectionFunctor* functor1 = (*mCompositeProjectionFunctor)[index + 1];
      
#if 0
      std::cout << " tree level " << treeLevel << " using functors " << functor0->to_string() << " " << functor1->to_string() << std::endl;
      std::cout << "launch domain at tree level " << treeLevel
      << launchDomain << std::endl;
#endif
      
      ArgumentMap argMap;
      CompositeArguments args = { imageSize, depthFunc, blendFuncSource, blendFuncDestination, blendEquation };
      IndexTaskLauncher treeCompositeLauncher(compositeTaskID, launchDomain, TaskArgument(&args, sizeof(args)), argMap);
      
      RegionRequirement req0(sourceFragmentPartition, functor0->id(), READ_WRITE, EXCLUSIVE, image);
      addImageFieldsToRequirement(req0);
      treeCompositeLauncher.add_region_requirement(req0);
      
      RegionRequirement req1(sourceFragmentPartition, functor1->id(), READ_ONLY, EXCLUSIVE, image);
      addImageFieldsToRequirement(req1);
      treeCompositeLauncher.add_region_requirement(req1);
      
      FutureMap futures = runtime->execute_index_space(context, treeCompositeLauncher);
      
      if(treeLevel > 1) {
              
        futures = launchTreeReduction(imageSize, treeLevel - 1, depthFunc, blendFuncSource, blendFuncDestination, blendEquation, compositeTaskID, sourceFragmentPartition, image, runtime, context, nodeID, maxTreeLevel);
      }
      return futures;
      
    }
    
    
    
    FutureMap ImageReduction::reduceAssociative() {
      int maxTreeLevel = numTreeLevels(mImageSize);
      return launchTreeReduction(mImageSize, maxTreeLevel, mDepthFunction, mGlBlendFunctionSource, mGlBlendFunctionDestination, mGlBlendEquation,
                                 mCompositeTaskID, mSourceFragmentPartition, mSourceImage,
                                 mRuntime, mContext, mLocalCopyOfNodeID, maxTreeLevel);
    }
    
    
    FutureMap ImageReduction::reduce_associative_commutative(){
      return reduceAssociative();
    }
    
    FutureMap ImageReduction::reduce_associative_noncommutative(){
      if(mNumSimulationBounds == mImageSize.numImageLayers) {
        return reduceAssociative();
      } else {
        std::cout << "cannot reduce noncommutatively until simulation bounds are provided" << std::endl;
        std::cout << "call ImageReduction::preregisterSimulationBounds(SimulationBoundsCoordinate *bounds, int numBounds) before starting Legion runtime" << std::endl;
        return FutureMap();
      }
    }
    
    
    
    FutureMap ImageReduction::launchPipelineReduction() {
      return FutureMap();
    }
    
    
    
    FutureMap ImageReduction::reduceNonassociative() {
      return launchPipelineReduction();
    }
    
    FutureMap ImageReduction::reduce_nonassociative_commutative(){
      return reduceNonassociative();
    }
    
    FutureMap ImageReduction::reduce_nonassociative_noncommutative(){
      return reduceNonassociative();
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
      create_image_field_pointers(args.imageSize, displayPlane, r, g, b, a, z, userdata, stride, runtime, ctx);
      
      FILE *outputFile = fopen(outputFileName.c_str(), "wb");
      fwrite(r, numPixelFields * sizeof(*r), args.imageSize.pixelsPerLayer(), outputFile);
      fclose(outputFile);
      
      display.stop();
      cout << display.to_string() << endl;
    }
    
    
    
    Future ImageReduction::display(int t) {
      DisplayArguments args = { mImageSize, t };
      TaskLauncher taskLauncher(mDisplayTaskID, TaskArgument(&args, sizeof(args)));
      DomainPoint origin = DomainPoint(Point<image_region_dimensions>::ZEROES());
      LogicalRegion displayPlane = mRuntime->get_logical_subregion_by_color(mDepthPartition, origin);
      RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      taskLauncher.add_region_requirement(req);
      Future displayFuture = mRuntime->execute_task(mContext, taskLauncher);
      return displayFuture;
    }
    
  }
}
