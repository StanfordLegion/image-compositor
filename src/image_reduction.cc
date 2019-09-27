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
#include "image_reduction.h"

#include "mappers/default_mapper.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>


using namespace LegionRuntime::Accessor;

/******************************************************************************
 
 Theory of operation.
 
 The image compositor maintains an Image 3D logical region.  This region is
 partitioned in slices like a loaf of bread where each slice is an image.
 
 The framework supports two constructors, an older constructor that is used
 mainly for testing and a newer constructor that takes a simulation logical
 partition as an argument.  We expect that applications will use the newer
 constructor.
 
 An application simulates, renders, and composites.  The framework is not
 involved with the simulation.  The framework provides access to the necessary
 logical partitions to index launch render operations, but those operations and
 the corresponding render tasks are written by the simulation developer.  See
 the README for a documented example.  After rendering is complete the
 application requests the framework to composite the resulting images.
 
 In order to ensure that compositing produces a correct result in the presence
 of non-commutative blending operations the following algorithm is followed.
 If the newer constructor is used this algorithm is followed regardless of the
 type of compositing operator, as it is valid for commutative depth comparisons
 as well as for blending.
 
 The framework constructs a new logical partition of the Image region that will
 be used only during the render step.  Call this the render image partition.
 The render image partition has a permuted color space which has the effect of
 permuting the individual images in the image logical region.  As a result of
 this the render tasks will write their output into the image region in a
 permuted order.
 
 In order to construct the render image partition the framework inserts all of
 the subregions from the simulation logical partition into a KD tree.  Each KD
 tree entry is indexed by the simulation coordinates.  Each entry also stores
 the coordinates of the corresponding subregion of the image region in linear
 order from front to back.  To construct the rendering image partition the
 framework traverses the KD tree in its natural order and retrieves the
 coordinates of the corresponding image region.  These coordinates are used to
 populate a new rendering color space which is a permutation of the original
 image color space.  The framework uses this new color space to create a
 render image partition in which the subregions are ordered by KD tree traversal
 order rather than by the original linear order.
 
 The framework also constructs a composite logical partition that is used for
 the index task launch of the composite step.  This partition orders the images
 linearly as in the loaf of bread analogy.  The composite projection functors
 composite these images in a binary tree structure.  Since the images have been
 written to the image region by the render step in correct compositing order
 the resulting composite will be correct even in the present of blending
 operators.
 
 Note that is the first constructor is used then the framework will use the
 composite logical partition for both rendering and compositing.  This is only
 correct for commutative compositing operators like depth comparison.
 
 
 
 @aheirich I've not read the text above yet, but please work out an example on paper using a 2-d grid with integer coordinates, an image plane volume with 3-d coordinates, a source partition color space using roman letters as color names (i.e. 'a', 'b', 'c', 'd') and a compositing partition color space using greek letters (i.e. alpha, beta, gamma, ...)
 
 the goal is to produce a partition of the image plane volume with a color space made up of roman letters that is consistent with the compositing space partition (e.g. if subvolumes S4 and S12 are the first two leaves of the k-d tree and their colors in the source partition color space are 'b' and 'e', then the image plane colored 'b' in the synthesized partition must be the same image plane colored 'alpha' in the compositing partitiong, and the image plane colored 'e' in the syntheisized partition must be the same image plane colored 'beta' in hte compositing partition
 e.g. if S4 and S12 are simulation subregions corresponding to the first two leaves of the KD tree;
 the colors of S4 and S12 are b and d;
 the render image partition has subregions I1, I2 with colors b and d;
 renderImage[b] is the same subregion as compositeImage[alpha]
 renderImage[d] is the same subregion as compositeImage[beta]
 I think
 
 there are at least two ways to achieve this based on a traversal of the k-d tree
 and as is often the case in a strongly-typed world, if you can even construct a partition with the right types (i.e. colors that are roman letters and subvolumes that are image planes), you're most of the way there
 
 Simulation logical region, index space in 2D = { 0,0 0,1 1,0 1,1 }
 Image logical region, index space in 3D { 0,0,0 0,0,1 0,0,2 0,0,3 }
 Simulation partition, color space { a, b, c, d }
 Image compositing partition, color space { alpha, beta, gamma, delta }
 
 constructing renderImage:
 put the simulation coordinates in the KD tree.  For simplicity assume the simulation
 coordinates are the same as the index space coordinates.  WLOG assume the KD tree
 orders the coordinates { 0,1 1,1 0,0 1,0 }.  This corresponds to an ordering of the
 simulation partition color space { b, d, a, c }.
 now construct a renderImage partition where { b, d, a, c } corresponds to
 { alpha, beta, gamma, delta } in compositeImage[].  How?
 
 
 
 ******************************************************************************/


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
    KDTree<image_region_dimensions, long long int>* mKDTree;
    
    
    /**
     * Use this constructor with your simulation partition.
     **/
    ImageReduction::ImageReduction(LogicalPartition partition, ImageDescriptor imageDescriptor, Context context, HighLevelRuntime *runtime, MapperID mapperID) {
      Domain domain = runtime->get_index_partition_color_space(context, partition.get_index_partition());
      imageDescriptor.simulationLogicalPartition = partition;
      imageDescriptor.simulationColorSpace = 
        runtime->get_index_partition_color_space(context, partition.get_index_partition());
      imageDescriptor.simulationDomain = domain;
      imageDescriptor.hasPartition = true;
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

      createImageRegion(mSourceIndexSpace, mSourceImage, mSourceImageDomain, mSourceImageFields, fieldID, context);
      partitionImageByImageDescriptor(mSourceImage, context, runtime, imageDescriptor);
      partitionImageByKDTree(mSourceImage, partition, context, runtime, imageDescriptor);
      
      initializeNodes(runtime, context);
      
      assert(mNodeID != -1);
      initializeViewMatrix();
      createTreeDomains(mNodeID, numTreeLevels(imageDescriptor), runtime, imageDescriptor);
      
    }
    
    /**
     * use this constructor for testing and applications that don't have a simulation partition.
     **/
    ImageReduction::ImageReduction(ImageDescriptor imageDescriptor, Context context, HighLevelRuntime *runtime, MapperID mapperID) {
      imageDescriptor.hasPartition = false;
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

      createImageRegion(mSourceIndexSpace, mSourceImage, mSourceImageDomain, mSourceImageFields, fieldID, context);
      partitionImageByImageDescriptor(mSourceImage, context, runtime, imageDescriptor);
      mRenderImageColorSpace = mCompositeImageColorSpace;
      mRenderImageDomain = mCompositeImageDomain;
      mRenderImagePartition = mCompositeImagePartition;

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
    
    
    void ImageReduction::createImageRegion(IndexSpace& indexSpace, LogicalRegion &region, Domain &domain, FieldSpace& fields, legion_field_id_t fieldID[], Context context) {
      Point<image_region_dimensions> p0 = mImageDescriptor.origin();
      Point <image_region_dimensions> p1 = mImageDescriptor.upperBound() - Point<image_region_dimensions>::ONES();
      Rect<image_region_dimensions> imageBounds(p0, p1);
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
    
    
    void ImageReduction::partitionImageByImageDescriptor(LogicalRegion image, Context ctx, HighLevelRuntime* runtime, ImageDescriptor imageDescriptor) {
      Point<image_region_dimensions> p0;
      p0 = mImageDescriptor.origin();
      Point <image_region_dimensions> p1 = mImageDescriptor.numLayers() - Point<image_region_dimensions>::ONES();
      Rect<image_region_dimensions> color_bounds(p0, p1);
      IndexSpace colorIndexSpace = runtime->create_index_space(ctx, color_bounds);
      IndexSpace is_parent = image.get_index_space();
      Transform<image_region_dimensions, image_region_dimensions> identity;
      for(unsigned i = 0; i < image_region_dimensions; ++i) {
        for(unsigned j = 0; j < image_region_dimensions; ++j) identity[i][j] = 0;
        identity[i][i] = 1;
      }
      Point<image_region_dimensions> p2 = imageDescriptor.layerSize()
      - Point<image_region_dimensions>::ONES();
      Rect<image_region_dimensions> slice(p0, p2);
      IndexPartition ip = runtime->create_partition_by_restriction(ctx,
                           is_parent, colorIndexSpace, identity, slice);
      mCompositeImagePartition = runtime->get_logical_partition(ctx, image, ip);
      runtime->attach_name(mCompositeImagePartition, "compositeImagePartition");
      mCompositeImageDomain = runtime->get_index_space_domain(ctx, colorIndexSpace);
    }
    
    /*
     * KDTree-based compositing
     
     Note: can we stop returning legion data structures from the init and make them all accessible through the image-compositor?
     
     
     Case 1: no partition
     Construct old linear image partition
     Copy it to render partition
     Create an accessor for the render partition
     Change the name of the existing accessor to “CompositePartition”
     
     Case 2: partition
     Construct old linear image partition
     Construct new resorted render partition
     
     Modify render.cc to access new compositor and render partitions
     In example_2 and soleil-x
     
     To construct the new render partition do this
     The goal is to construct a partition with a permuted color map so that the rendered results will be placed into the correct positions in the compositor region.
     We reorder the simulation subregions according to a KD Tree.
     This gives a reordered color space according to the KD Tree traversal.
     (We form a new partition of the image region using the source partition's color space and the position of the simulation volume within the kd tree to determine which image plane that volume does its volume rendering to.)
     We reorder the entries of the source partition color space so that they correspond to the KDTree traversal order.   Then we repartition the image plane using that color space.  This will make the render tasks write to the permuted image region, so they are in correct compositing order.
     sean: "the kd-tree is actually defining a mapping from image plane number -> color in simulation volume, but what we are capturing in this second image plane partition is the inverse: color in simulation volume back to image plane number"
     
     When we index launch the render tasks we launch them in order according to the simulation partition.
     We make a region requirement for the simulation subdomain, and another for the image render subdomain.
     The image render partition is ordered according to the KD Tree traversal.
     The simulation partition still has its original ordering so each render task will read from a successive simulation subdomain, but will write to a permuted location in the image region.
     After rendering we launch the composite task, using the old image partition, and the existing projection functors.
     The projection functors know how to work with this partition.

     */
    
    void ImageReduction::partitionImageByKDTree(LogicalRegion image, LogicalPartition sourcePartition, Context ctx, HighLevelRuntime* runtime, ImageDescriptor imageDescriptor) {
      mRenderImageColorSpace = imageDescriptor.simulationColorSpace;
      buildKDTree(imageDescriptor, ctx, runtime);
      Legion::DomainPoint *coloring = new Legion::DomainPoint[mKDTree->size()];
      mKDTree->getColorMap(coloring);
      DomainColoring domainColoring;
      for(unsigned i = 0; i < mKDTree->size(); ++i) {
        domainColoring[i] = coloring[i];
      }
      delete [] coloring;
      IndexPartition ip = runtime->create_index_partition(ctx, mSourceImage.get_index_space(), mSourceImageDomain, domainColoring, true);
      LogicalPartition mRenderImagePartition = runtime->get_logical_partition(ctx, mSourceImage, ip);
    }

    
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
    
    
    void ImageReduction::buildKDTree(ImageDescriptor imageDescriptor,
                                     Context ctx,
                                     HighLevelRuntime *runtime) {
      Rect<image_region_dimensions> rect = imageDescriptor.simulationDomain;
      KDTreeValue* elements = new KDTreeValue[rect.volume()];
      unsigned index = 0;
      for(Domain::DomainPointIterator it(imageDescriptor.simulationDomain); it; it++) {
        DomainPoint color(it);
        IndexSpace subregion = runtime->get_index_subspace(ctx, imageDescriptor.simulationLogicalPartition.get_index_partition(), color);
        Domain subdomain = runtime->get_index_space_domain(ctx, subregion);
        LegionRuntime::Arrays::Rect<image_region_dimensions> rect = subdomain.get_rect<image_region_dimensions>();
        KDTreeValue value = { rect, color };
        elements[index++] = value;
      }
      
      mKDTree = new KDTree<image_region_dimensions, long long int>(elements, rect.volume());
      delete [] elements;
      char buffer[256];
      std::cout << gethostname(buffer, sizeof(buffer)) << " pid " << getpid() << " built a KDTree and colormap with " << rect.volume() << " entries" << std::endl;
    }
    
    
    void ImageReduction::initial_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, HighLevelRuntime *runtime) {
      
#ifdef TRACE_TASKS
      std::cout << describe_task(task) << std::endl;
#endif
      ImageDescriptor imageDescriptor = *((ImageDescriptor*)task->args);
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
      if(imageDescriptor.hasPartition) {
        buildKDTree(imageDescriptor, ctx, runtime);
      }
    }
    
    
    void ImageReduction::initializeNodes(HighLevelRuntime* runtime, Context context) {
      launch_task_compositeImage(mInitialTaskID, runtime, context, NULL, 0, true);
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
    
    
    FutureMap ImageReduction::launch_task_compositeImage(unsigned taskID, HighLevelRuntime* runtime, Context context, void *args, int argLen, bool blocking){
      
      ArgumentMap argMap;
      int totalArgLen = sizeof(mImageDescriptor) + argLen;
      char *argsBuffer = new char[totalArgLen];
      memcpy(argsBuffer, &mImageDescriptor, sizeof(mImageDescriptor));
      if(argLen > 0) {
        memcpy(argsBuffer + sizeof(mImageDescriptor), args, argLen);
      }
      
      // if imageDescriptor has a partition launch over the partition
      // otherwise launch over the image compositeImageDomain
      Domain domain;
      if(mImageDescriptor.hasPartition) {
        domain = mImageDescriptor.simulationDomain;
      } else {
        domain = mCompositeImageDomain;
      }
      
      IndexTaskLauncher compositeImageLauncher(taskID, domain, TaskArgument(argsBuffer, totalArgLen), argMap, Predicate::TRUE_PRED, false, mMapperID);
      RegionRequirement req(mCompositeImagePartition, 0, READ_WRITE, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      compositeImageLauncher.add_region_requirement(req);
      
      FutureMap futures = runtime->execute_index_space(context, compositeImageLauncher);
      
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
      
      compositeFunction = ImageReductionComposite::compositeFunctionPointer(args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination, args.blendEquation);
      compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0, args.imageDescriptor.pixelsPerLayer(), stride0, stride1);
      //      composite.stop();
      //      std::cout << composite.to_string() << std::endl;

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
                                   mCompositeTaskID, mCompositeImagePartition, mSourceImage,
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
      LogicalRegion displayPlane = mRuntime->get_logical_subregion_by_color(mCompositeImagePartition, origin);
      RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE, mSourceImage);
      addImageFieldsToRequirement(req);
      taskLauncher.add_region_requirement(req);
      Future displayFuture = mRuntime->execute_task(context, taskLauncher);
      return displayFuture;
    }
    
  }
}
