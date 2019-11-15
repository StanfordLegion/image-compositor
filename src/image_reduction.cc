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

using namespace std;
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

 In order to ensure that compositing produces a correct result the following
 algorithm is followed.  This is valid for commutative depth comparisons
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


 ******************************************************************************/


namespace Legion {
  namespace Visualization {


    // declare module static data

    std::vector<ImageReduction::CompositeProjectionFunctor*> *ImageReduction::mCompositeProjectionFunctor = NULL;
    std::vector<Domain> *ImageReduction::mHierarchicalTreeDomain = nullptr;
    GLfloat ImageReduction::mGlViewTransform[numMatrixElements4x4];
    ImageReduction::PixelField ImageReduction::mGlConstantColor[numPixelFields];
    GLenum ImageReduction::mGlBlendEquation;
    GLenum ImageReduction::mGlBlendFunctionSource;
    GLenum ImageReduction::mGlBlendFunctionDestination;
    TaskID ImageReduction::mInitialTaskID;
    TaskID ImageReduction::mCompositeTaskID;
    TaskID ImageReduction::mDisplayTaskID;
    KDTree<image_region_dimensions, long long int>* ImageReduction::mSimulationKDTree = nullptr;
    KDTree<image_region_dimensions, long long int>* ImageReduction::mImageKDTree = nullptr;


    /**
     * Use this constructor with your simulation partition.
     **/
    ImageReduction::ImageReduction(
      LogicalRegion region,
      LogicalPartition partition,
      legion_field_id_t pFields[],
      int numPFields,
      ImageDescriptor imageDescriptor,
      Context context,
      HighLevelRuntime *runtime) {
      Domain domain = runtime->get_index_partition_color_space(context, partition.get_index_partition());
      imageDescriptor.simulationLogicalRegion = region;
      imageDescriptor.simulationLogicalPartition = partition;
      imageDescriptor.numPFields = numPFields;
      assert(numPFields <= Legion::Visualization::max_pFields);
      memcpy(imageDescriptor.pFields, pFields, numPFields * sizeof(legion_field_id_t));
      imageDescriptor.simulationColorSpace =
        runtime->get_index_partition_color_space(context, partition.get_index_partition());
      imageDescriptor.simulationDomain = domain;
      imageDescriptor.hasPartition = true;
      imageDescriptor.numImageLayers = domain.get_volume();
      mImageDescriptor = imageDescriptor;
      mRuntime = runtime;
      mDepthFunction = 0;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mRenderImageDomain = imageDescriptor.simulationDomain;

      mGlBlendEquation = GL_FUNC_ADD;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mDepthFunction = 0;
      legion_field_id_t fieldID[6];
      createImageRegion(mSourceIndexSpace, mSourceImage, mSourceImageDomain, mSourceImageFields, fieldID, context);
      partitionImageByImageDescriptor(mSourceImage, context, runtime, imageDescriptor);
      initializeNodes(mRuntime, context);
      createProjectionFunctors(runtime, imageDescriptor.numImageLayers);
      partitionImageByKDTree(mSourceImage, partition, context, runtime, imageDescriptor);
      initializeViewMatrix();
      createTreeDomains(numTreeLevels(imageDescriptor), runtime, imageDescriptor);
    }

    /**
     * use this constructor for testing and applications that don't have a simulation partition.
     **/
    ImageReduction::ImageReduction(ImageDescriptor imageDescriptor, Context context, HighLevelRuntime *runtime) {
      imageDescriptor.hasPartition = false;
      mImageDescriptor = imageDescriptor;
      mRuntime = runtime;
      mDepthFunction = 0;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;

      mGlBlendEquation = GL_FUNC_ADD;
      mGlBlendFunctionSource = 0;
      mGlBlendFunctionDestination = 0;
      mDepthFunction = 0;
      legion_field_id_t fieldID[6];

      createImageRegion(mSourceIndexSpace, mSourceImage, mSourceImageDomain, mSourceImageFields, fieldID, context);
      partitionImageByImageDescriptor(mSourceImage, context, runtime, imageDescriptor);
      initializeNodes(mRuntime, context);
      createProjectionFunctors(runtime, imageDescriptor.numImageLayers);
      mRenderImageColorSpace = mCompositeImageColorSpace;
      mRenderImageDomain = mCompositeImageDomain;
      mRenderImagePartition = mCompositeImagePartition;

      initializeViewMatrix();
      createTreeDomains(numTreeLevels(imageDescriptor), runtime, imageDescriptor);
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
      mRuntime->attach_name(region, "sourceImage");
      fieldID[0] = FID_FIELD_R;
      fieldID[1] = FID_FIELD_G;
      fieldID[2] = FID_FIELD_B;
      fieldID[3] = FID_FIELD_A;
      fieldID[4] = FID_FIELD_Z;
      fieldID[5] = FID_FIELD_USERDATA;
      // fill the region initially with ZEROES
      PixelField zero = 0;
      TaskArgument arg(&zero, sizeof(zero));
      FillLauncher fillLauncher(region, region, arg);
      fillLauncher.add_field(FID_FIELD_R);
      fillLauncher.add_field(FID_FIELD_G);
      fillLauncher.add_field(FID_FIELD_B);
      fillLauncher.add_field(FID_FIELD_A);
      fillLauncher.add_field(FID_FIELD_Z);
      fillLauncher.add_field(FID_FIELD_USERDATA);
      mRuntime->fill_fields(context, fillLauncher);
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

    void ImageReduction::partitionImageByKDTree(LogicalRegion image,
      LogicalPartition sourcePartition, Context ctx, HighLevelRuntime* runtime, ImageDescriptor imageDescriptor) {
      mRenderImageColorSpace = imageDescriptor.simulationColorSpace;
      Legion::Point<image_region_dimensions> *coloring = new Legion::Point<image_region_dimensions>[mSimulationKDTree->size()];
      mSimulationKDTree->getColorMap(coloring);

      // create a logical region to hold the coloring and extent
      Point<image_region_dimensions> p0 = mImageDescriptor.origin();
      Point <image_region_dimensions> p1 = mImageDescriptor.upperBound() - Point<image_region_dimensions>::ONES();
      Rect<image_region_dimensions> imageBounds(p0, p1);
      IndexSpace coloringIndexSpace = mRuntime->create_index_space(ctx, mRenderImageColorSpace);
      FieldSpace coloringFields = mRuntime->create_field_space(ctx);
      mRuntime->attach_name(coloringFields, "render image coloring fields");

      FieldAllocator coloringAllocator = mRuntime->create_field_allocator(ctx, coloringFields);
      FieldID fidColor = coloringAllocator.allocate_field(sizeof(Point<image_region_dimensions>), FID_FIELD_COLOR);
      assert(fidColor == FID_FIELD_COLOR);
      FieldID fidExtent = coloringAllocator.allocate_field(sizeof(Rect<image_region_dimensions>), FID_FIELD_EXTENT);
      assert(fidExtent == FID_FIELD_EXTENT);

      LogicalRegion coloringExtentRegion = mRuntime->create_logical_region(ctx, coloringIndexSpace, coloringFields);

      // write the color and extent values into the region
      RegionRequirement coloringReq(coloringExtentRegion, WRITE_DISCARD, EXCLUSIVE, coloringExtentRegion);
      coloringReq.add_field(FID_FIELD_COLOR);
      coloringReq.add_field(FID_FIELD_EXTENT);
      InlineLauncher coloringLauncher(coloringReq);
      PhysicalRegion coloringPhysicalRegion = mRuntime->map_region(ctx, coloringLauncher);
      const FieldAccessor<WRITE_DISCARD, Point<image_region_dimensions>,
        image_region_dimensions, long long int,
        Realm::AffineAccessor<Point<image_region_dimensions>, image_region_dimensions, long long int> >
        acc_color(coloringPhysicalRegion, FID_FIELD_COLOR);

      const FieldAccessor<WRITE_DISCARD, Rect<image_region_dimensions>,
        image_region_dimensions, long long int,
        Realm::AffineAccessor<Rect<image_region_dimensions>, image_region_dimensions, long long int> >
        acc_extent(coloringPhysicalRegion, FID_FIELD_EXTENT);

      Rect<image_region_dimensions> rect = imageBounds;

      for(unsigned i = 0; i < mSimulationKDTree->size(); ++i) {
        rect.lo.z = rect.hi.z = i;
        acc_extent[coloring[i]] = rect;
        acc_color[coloring[i]] = coloring[i];
      }
      // partition the coloring region by field
      IndexPartition coloringIP = mRuntime->create_partition_by_field(ctx,
        coloringExtentRegion, coloringExtentRegion, FID_FIELD_COLOR, coloringIndexSpace);
      LogicalPartition coloringPartition = runtime->get_logical_partition(ctx, coloringExtentRegion, coloringIP);
      IndexPartition renderImageIP = mRuntime->create_partition_by_image_range(
        ctx, mSourceIndexSpace, coloringPartition, coloringExtentRegion, FID_FIELD_EXTENT, coloringIndexSpace);
      mRenderImagePartition = runtime->get_logical_partition(ctx, mSourceImage, renderImageIP);
      mRuntime->attach_name(mRenderImagePartition, "render image partition");
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


    void ImageReduction::createProjectionFunctors(Runtime* runtime, int numImageLayers) {

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


    void ImageReduction::buildKDTrees(ImageDescriptor imageDescriptor,
                                     Context ctx,
                                     HighLevelRuntime *runtime) {
      Rect<image_region_dimensions> rect = imageDescriptor.simulationDomain;
      KDTreeValue* simulationElements = new KDTreeValue[rect.volume()];
      unsigned index = 0;
      Point<image_region_dimensions> p0 = Point<image_region_dimensions>::ZEROES();
      Rect<image_region_dimensions> zeroRect(p0, p0);

      for(Domain::DomainPointIterator it(imageDescriptor.simulationDomain); it; it++) {
        DomainPoint color(it.p);
        IndexSpace subregion = runtime->get_index_subspace(ctx,
          imageDescriptor.simulationLogicalPartition.get_index_partition(), color);
        Domain subdomain = runtime->get_index_space_domain(ctx, subregion);
        Legion::Rect<image_region_dimensions> simulationRect(color, color);
        KDTreeValue simulationValue;
        simulationValue.extent = simulationRect;
        simulationValue.color = color;
        simulationValue.extent2 = zeroRect;
        simulationElements[index] = simulationValue;
        index++;
      }

      mSimulationKDTree = new KDTree<image_region_dimensions, long long int>(simulationElements, rect.volume());
      delete [] simulationElements;

      Rect<image_region_dimensions>* simulationExtents = new Rect<image_region_dimensions>[rect.volume()];
      mSimulationKDTree->getExtent(simulationExtents);
      KDTreeValue* imageElements = new KDTreeValue[rect.volume()];

      for(unsigned i = 0; i < rect.volume(); ++i) {
        Point<image_region_dimensions> p0(0, 0, i);
        Point<image_region_dimensions> p1 = imageDescriptor.upperBound() -
          Point<image_region_dimensions>::ONES();
        p1[image_region_dimensions - 1] = i;
        Rect<image_region_dimensions> imageRect(p0, p1);
        KDTreeValue imageElement;
        imageElement.extent = imageRect;
        imageElement.color = Point<image_region_dimensions>::ZEROES();
        imageElement.extent2 = simulationExtents[i];
        imageElements[i] = imageElement;
      }

      delete [] simulationExtents;
      mImageKDTree = new KDTree<image_region_dimensions, long long int>(imageElements, rect.volume());
      delete [] imageElements;
    }


    void ImageReduction::initial_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, HighLevelRuntime *runtime) {

#ifdef TRACE_TASKS
      std::cout << describe_task(task) << std::endl;
#endif
      ImageDescriptor imageDescriptor = *((ImageDescriptor*)task->args);
      createProjectionFunctors(runtime, imageDescriptor.numImageLayers);
      if(imageDescriptor.hasPartition) {
        buildKDTrees(imageDescriptor, ctx, runtime);
      }
    }



    void ImageReduction::initializeViewMatrix() {
      memset(mGlViewTransform, 0, sizeof(mGlViewTransform));
      mGlViewTransform[0] = mGlViewTransform[5] = mGlViewTransform[10] = mGlViewTransform[15] = 1.0f;
    }


    void ImageReduction::createTreeDomains(int numTreeLevels, Runtime* runtime, ImageDescriptor imageDescriptor) {
      if(mHierarchicalTreeDomain == NULL) {
        mHierarchicalTreeDomain = new std::vector<Domain>();
      }

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
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t,
        Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_r(region, FID_FIELD_R);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t,
        Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_g(region, FID_FIELD_G);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t,
        Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_b(region, FID_FIELD_B);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t,
        Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_a(region, FID_FIELD_A);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t,
        Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_z(region, FID_FIELD_Z);
        const FieldAccessor<READ_ONLY, PixelField, image_region_dimensions, coord_t,
        Realm::AffineAccessor<PixelField, image_region_dimensions, coord_t> > acc_userdata(region, FID_FIELD_USERDATA);
        r = (PixelField*)acc_r.ptr(rect, stride[0]);
        g = (PixelField*)acc_g.ptr(rect, stride[1]);
        b = (PixelField*)acc_b.ptr(rect, stride[2]);
        a = (PixelField*)acc_a.ptr(rect, stride[3]);
        z = (PixelField*)acc_z.ptr(rect, stride[4]);
        userdata = (PixelField*)acc_userdata.ptr(rect, stride[5]);

      }

    }


    void ImageReduction::initializeNodes(HighLevelRuntime* runtime, Context context) {
      unsigned taskID = mInitialTaskID;
      ArgumentMap argMap;
      int totalArgLen = sizeof(mImageDescriptor);
      char *argsBuffer = new char[totalArgLen];
      memcpy(argsBuffer, &mImageDescriptor, sizeof(mImageDescriptor));

      // if imageDescriptor has a partition launch over the partition
      // otherwise launch over the image compositeImageDomain
      Domain domain;
      LogicalPartition partition;
      LogicalRegion region;
      if(mImageDescriptor.hasPartition) {
        domain = mImageDescriptor.simulationDomain;
        partition = mImageDescriptor.simulationLogicalPartition;
        region = mImageDescriptor.simulationLogicalRegion;
      } else {
        domain = mCompositeImageDomain;
        //partition =
//TODO create mCompositeImagePartition in createImageRegion
        region = mSourceImage;
      }

      IndexTaskLauncher launcher(taskID, domain,
        TaskArgument(argsBuffer, totalArgLen), argMap, Predicate::TRUE_PRED, false);
      RegionRequirement req(partition, 0, READ_ONLY, EXCLUSIVE, region);
      if(mImageDescriptor.hasPartition) {
        for(int i = 0; i < mImageDescriptor.numPFields; ++i) {
          req.add_field(mImageDescriptor.pFields[i]);
        }
      } else {
        addImageFieldsToRequirement(req);
      }
      launcher.add_region_requirement(req);
      FutureMap futures = runtime->execute_index_space(context, launcher);
      futures.wait_all_results();
      delete [] argsBuffer;
      if(mImageDescriptor.hasPartition) {
        buildKDTrees(mImageDescriptor, context, runtime);
      }
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

    KDNode<image_region_dimensions, long long int>*
    ImageReduction::findFragmentInKDTree(PhysicalRegion fragment) {
      Legion::DomainT<image_region_dimensions, long long int> domain =
      fragment.get_bounds<image_region_dimensions, long long int>();
      //Legion::Rect<image_region_dimensions> rect = domain.Rect<image_region_dimensions, long long int>();
      Legion::Rect<image_region_dimensions> rect = (Domain)domain;
      KDTreeValue value;
      value.extent = rect;
      KDNode<image_region_dimensions, long long int>* imageNode = mImageKDTree->find(value);
      value.extent = imageNode->mValue.extent2;
      KDNode<image_region_dimensions, long long int>* simulationNode = mSimulationKDTree->find(value);
      return simulationNode;
    }


    bool ImageReduction::flipRegions(PhysicalRegion fragment0,
                                     PhysicalRegion fragment1,
                                     float cameraDirection[image_region_dimensions]) {
      if(mSimulationKDTree == nullptr) return false;
      if(cameraDirection == nullptr) return false;
      KDNode<image_region_dimensions, long long int>* node0 = findFragmentInKDTree(fragment0);
      KDNode<image_region_dimensions, long long int>* node1 = findFragmentInKDTree(fragment1);
      unsigned axis0 = node0->mLevel % image_region_dimensions;
      unsigned axis1 = node1->mLevel % image_region_dimensions;
      float splittingPlaneNormal[image_region_dimensions] = { 0 };
      if(axis0 == axis1) {
        if(axis0 == 0) splittingPlaneNormal[1] = 1;
        else splittingPlaneNormal[0] = 1;
      } else {
        unsigned axisSum = axis0 + axis1;
        switch(axisSum) {
          case 1: splittingPlaneNormal[2] = 1; break;
          case 2: splittingPlaneNormal[1] = 1; break;
          case 3: splittingPlaneNormal[0] = 1; break;
        }
      }
      float dot = 0;
      for(unsigned i = 0; i < image_region_dimensions; ++i) {
        dot += splittingPlaneNormal[i] * cameraDirection[i];
      }
      return dot < 0;
    }


    void ImageReduction::composite_task(const Task *task,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context ctx,
                                        HighLevelRuntime *runtime) {
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
      compositeFunction = ImageReductionComposite::compositeFunctionPointer(
        args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination, args.blendEquation);
      create_image_field_pointers(args.imageDescriptor, fragment0,
        r0, g0, b0, a0, z0, userdata0, stride0, runtime, ctx, true);
      create_image_field_pointers(args.imageDescriptor, fragment1,
        r1, g1, b1, a1, z1, userdata1, stride1, runtime, ctx, false);

#define SHOW_COMPOSITING 0
#if SHOW_COMPOSITING
ImageReduction::PixelField rr0 = *r0;
ImageReduction::PixelField gg0 = *g0;
ImageReduction::PixelField bb0 = *b0;
ImageReduction::PixelField aa0 = *a0;
ImageReduction::PixelField zz0 = *z0;
ImageReduction::PixelField uu0 = *userdata0;
ImageReduction::PixelField rr1 = *r1;
ImageReduction::PixelField gg1 = *g1;
ImageReduction::PixelField bb1 = *b1;
ImageReduction::PixelField aa1 = *a1;
ImageReduction::PixelField zz1 = *z1;
ImageReduction::PixelField uu1 = *userdata1;
#endif

      if(flipRegions(fragment0, fragment1, args.cameraDirection)) {
        compositeFunction(r1, g1, b1, a1, z1, userdata1, r0, g0, b0, a0, z0, userdata0,
          r0, g0, b0, a0, z0, userdata0, args.imageDescriptor.pixelsPerLayer(), stride1, stride0);
      } else {
        compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1,
          r0, g0, b0, a0, z0, userdata0, args.imageDescriptor.pixelsPerLayer(), stride0, stride1);
      }

#if SHOW_COMPOSITING
{
char buffer[256];
sprintf(buffer, "%s (%g %g %g %g %g) (%g %g %g %g %g) = %g %g %g %g %g\n",
__FUNCTION__, rr0, gg0, bb0, aa0, zz0, rr1, gg1, bb1, aa1, zz1, *r0, *g0, *b0, *a0, *z0);
std::cout << buffer;
}
#endif
      //      composite.stop();
      //      std::cout << composite.to_string() << std::endl;

    }





    FutureMap ImageReduction::launchTreeReduction(ImageDescriptor imageDescriptor, int treeLevel,
                                                  GLenum depthFunc, GLenum blendFuncSource,
                                                  GLenum blendFuncDestination, GLenum blendEquation,
                                                  int compositeTaskID, LogicalPartition sourcePartition,
                                                  LogicalRegion image,
                                                  Runtime* runtime, Context context,
                                                  int maxTreeLevel,
                                                  float cameraDirection[image_region_dimensions]) {
      Domain launchDomain = (*mHierarchicalTreeDomain)[treeLevel - 1];
      int index = (treeLevel - 1) * 2;
      CompositeProjectionFunctor* functor0 = (*mCompositeProjectionFunctor)[index];
      CompositeProjectionFunctor* functor1 = (*mCompositeProjectionFunctor)[index + 1];

      ArgumentMap argMap;
      CompositeArguments args;
      args.imageDescriptor = imageDescriptor;
      args.depthFunction = depthFunc;
      args.blendFunctionSource = blendFuncSource;
      args.blendFunctionDestination = blendFuncDestination;
      args.blendEquation = blendEquation;
      memcpy(args.cameraDirection, cameraDirection, sizeof(args.cameraDirection));
      IndexTaskLauncher treeCompositeLauncher(compositeTaskID, launchDomain,
        TaskArgument(&args, sizeof(args)), argMap, Predicate::TRUE_PRED, false);

      RegionRequirement req0(sourcePartition, functor0->id(), READ_WRITE, EXCLUSIVE, image);
      addImageFieldsToRequirement(req0);
      treeCompositeLauncher.add_region_requirement(req0);

      RegionRequirement req1(sourcePartition, functor1->id(), READ_ONLY, EXCLUSIVE, image);
      addImageFieldsToRequirement(req1);
      treeCompositeLauncher.add_region_requirement(req1);

      FutureMap futures = runtime->execute_index_space(context, treeCompositeLauncher);

      if(treeLevel > 1) {

        futures = launchTreeReduction(imageDescriptor, treeLevel - 1, depthFunc,
          blendFuncSource, blendFuncDestination, blendEquation, compositeTaskID,
          sourcePartition, image, runtime, context, maxTreeLevel, cameraDirection);
      }

      return futures;

    }



    FutureMap ImageReduction::reduceImages(Context context, float cameraDirection[]) {
      int maxTreeLevel = numTreeLevels(mImageDescriptor);
      if(maxTreeLevel > 0) {
        return launchTreeReduction(mImageDescriptor, maxTreeLevel, mDepthFunction,
          mGlBlendFunctionSource, mGlBlendFunctionDestination, mGlBlendEquation,
          mCompositeTaskID, mCompositeImagePartition, mSourceImage, mRuntime,
          context, maxTreeLevel, cameraDirection);
      } else {
        return FutureMap();
      }
    }



    void ImageReduction::display_task(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, HighLevelRuntime *runtime) {

#ifdef TRACE_TASKS
      std::cout << describe_task(task) << std::endl;
#endif
      DisplayArguments args = ((DisplayArguments*)task->args)[0];
      char fileName[1024];
      sprintf(fileName, "display.%d.tga", args.t);
      string outputFileName = string(fileName);
      PhysicalRegion displayPlane = regions[0];
      Stride stride;
      PixelField *r, *g, *b, *a, *z, *userdata;
      create_image_field_pointers(args.imageDescriptor, displayPlane, r, g, b, a, z, userdata, stride, runtime, ctx, false);

      FILE* f = fopen(fileName, (const char*)"w");
      if(f == nullptr) {
        std::cerr << "could not create file " << outputFileName << std::endl;
        return;
      }
      fputc (0x00, f);  /* ID Length, 0 => No ID   */
      fputc (0x00, f);  /* Color Map Type, 0 => No color map included   */
      fputc (0x02, f);  /* Image Type, 2 => Uncompressed, True-color Image */
      fputc (0x00, f);  /* Next five bytes are about the color map entries */
      fputc (0x00, f);  /* 2 bytes Index, 2 bytes length, 1 byte size */
      fputc (0x00, f);
      fputc (0x00, f);
      fputc (0x00, f);
      fputc (0x00, f);  /* X-origin of Image */
      fputc (0x00, f);
      fputc (0x00, f);  /* Y-origin of Image */
      fputc (0x00, f);
      fputc (args.imageDescriptor.width & 0xff, f);      /* Image Width */
      fputc ((args.imageDescriptor.width>>8) & 0xff, f);
      fputc (args.imageDescriptor.height & 0xff, f);     /* Image Height   */
      fputc ((args.imageDescriptor.height>>8) & 0xff, f);
      fputc (0x18, f);     /* Pixel Depth, 0x18 => 24 Bits  */
      fputc (0x20, f);     /* Image Descriptor  */
      fclose(f);

      f = fopen(fileName, (const char*)"ab");  /* reopen in binary append mode */

      for(int y = args.imageDescriptor.height - 1; y >= 0; y--) {
        for(int x = 0; x < args.imageDescriptor.width; ++x) {
          int index = x + y * args.imageDescriptor.width;
          GLubyte b_ = b[index] * 255;
          fputc(b_, f); /* write blue */
          GLubyte g_ = g[index] * 255;
          fputc(g_, f); /* write green */
          GLubyte r_ = r[index] * 255;
          fputc(r_, f);   /* write red */
        }
      }
      fclose(f);
      std::cout << "wrote image " << outputFileName << std::endl;

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
