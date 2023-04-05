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

std::vector<ImageReduction::CompositeProjectionFunctor*> *ImageReduction::mCompositeProjectionFunctor = nullptr;
std::vector<Domain> *ImageReduction::mHierarchicalTreeDomain = nullptr;
GLfloat ImageReduction::mGlViewTransform[numMatrixElements4x4];
ImageReduction::PixelField ImageReduction::mGlConstantColor[numPixelFields];
GLenum ImageReduction::mGlBlendEquation = 0;
GLenum ImageReduction::mGlBlendFunctionSource = 0;
GLenum ImageReduction::mGlBlendFunctionDestination = 0;
int mInitialTaskID = 0;
TaskID ImageReduction::mCompositeTaskID = 0;
TaskID ImageReduction::mDisplayTaskID = 0;
KDTree<image_region_dimensions, long long int>* ImageReduction::mSimulationKDTree = nullptr;
KDTree<image_region_dimensions, long long int>* ImageReduction::mImageKDTree = nullptr;
MapperID gMapperID = 0;

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
                               Runtime *runtime,
                               MapperID mapperID) {
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
  gMapperID = mapperID;

  mGlBlendEquation = GL_FUNC_ADD;
  mGlBlendFunctionSource = 0;
  mGlBlendFunctionDestination = 0;
  mDepthFunction = 0;
  legion_field_id_t fieldID[6];
  createImageRegion(mSourceIndexSpace, mSourceImage, mSourceImageDomain, mSourceImageFields, fieldID, context);
  createImagePartition(fieldID, context);
  partitionImageByImageDescriptor(mSourceImage, context, runtime, imageDescriptor);
__TRACE
  initializeNodes(mRuntime, context);
__TRACE
  createProjectionFunctors(runtime, imageDescriptor.numImageLayers);
__TRACE
  partitionImageByKDTree(mSourceImage, partition, context, runtime, imageDescriptor);
__TRACE
  initializeViewMatrix();
  createTreeDomains(numTreeLevels(imageDescriptor), runtime, imageDescriptor);
}

/**
 * use this constructor for testing and applications that don't have a simulation partition.
 **/
ImageReduction::ImageReduction(ImageDescriptor imageDescriptor,
                               Context context, Runtime *runtime,
                               MapperID mapperID) {
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
  gMapperID = mapperID;

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
    if(mInitialTaskID == 0)
      mInitialTaskID = Legion::Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(mInitialTaskID, "initial_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<initial_task>(registrar, "initial_task");
  }
  {
    if(mCompositeTaskID == 0)
      mCompositeTaskID = Legion::Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(mCompositeTaskID, "composite_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<composite_task>(registrar, "composite_task");
  }
  {
    if(mDisplayTaskID == 0)
      mDisplayTaskID = Legion::Runtime::generate_static_task_id();
    TaskVariantRegistrar registrar(mDisplayTaskID, "display_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<display_task>(registrar, "display_task");
  }
}


FieldSpace ImageReduction::imageFields(Context context) {
  FieldSpace fields = Legion::Runtime::get_runtime()->create_field_space(context);
  Legion::Runtime::get_runtime()->attach_name(fields, "pixel fields");
  {
    FieldAllocator allocator = Legion::Runtime::get_runtime()->create_field_allocator(context, fields);
    FieldID fidr = allocator.allocate_field(sizeof(PixelField), FID_FIELD_R);
    Legion::Runtime::get_runtime()->attach_name(fields, fidr, "R");
    assert(fidr == FID_FIELD_R);
    FieldID fidg = allocator.allocate_field(sizeof(PixelField), FID_FIELD_G);
    Legion::Runtime::get_runtime()->attach_name(fields, fidg, "G");
    assert(fidg == FID_FIELD_G);
    FieldID fidb = allocator.allocate_field(sizeof(PixelField), FID_FIELD_B);
    Legion::Runtime::get_runtime()->attach_name(fields, fidb, "B");
    assert(fidb == FID_FIELD_B);
    FieldID fida = allocator.allocate_field(sizeof(PixelField), FID_FIELD_A);
    Legion::Runtime::get_runtime()->attach_name(fields, fida, "A");
    assert(fida == FID_FIELD_A);
    FieldID fidz = allocator.allocate_field(sizeof(PixelField), FID_FIELD_Z);
    Legion::Runtime::get_runtime()->attach_name(fields, fidz, "Z");
    assert(fidz == FID_FIELD_Z);
    FieldID fidUserdata = allocator.allocate_field(sizeof(PixelField), FID_FIELD_USERDATA);
    Legion::Runtime::get_runtime()->attach_name(fields, fidUserdata, "USER");
    assert(fidUserdata == FID_FIELD_USERDATA);
  }
  return fields;
}


void ImageReduction::createImagePartition(legion_field_id_t fieldID[], Context context) {

  // create colors (0,0,0),(0,0,1)....(0,0,n-1)
  Point<image_region_dimensions> p0;
  p0 = mImageDescriptor.origin();
  Point <image_region_dimensions> p1 = mImageDescriptor.numLayers() - Point<image_region_dimensions>::ONES();
  Rect<image_region_dimensions> color_bounds(p0, p1);
  IndexSpace colorIndexSpace = mRuntime->create_index_space(context, color_bounds);

  // partition the image into slices (0,0,0),(w-1,h-1,0)
  IndexSpace is_parent = mSourceImage.get_index_space();
  Transform<image_region_dimensions, image_region_dimensions> identity;
  for(unsigned i = 0; i < image_region_dimensions; ++i) {
    for(unsigned j = 0; j < image_region_dimensions; ++j) identity[i][j] = 0;
    identity[i][i] = 1;
  }
  Point<image_region_dimensions> p2 = mImageDescriptor.layerSize()
  - Point<image_region_dimensions>::ONES();
  Rect<image_region_dimensions> slice(p0, p2);
  IndexPartition ip = mRuntime->create_partition_by_restriction(context,
         is_parent, colorIndexSpace, identity, slice, DISJOINT_COMPLETE_KIND);
  mCompositeImagePartition = mRuntime->get_logical_partition(context, mSourceImage, ip);
  mRuntime->attach_name(mCompositeImagePartition, "compositeImagePartition");
}


void ImageReduction::createImageRegion(IndexSpace& indexSpace, LogicalRegion &region, Domain &domain, FieldSpace& fields, legion_field_id_t fieldID[], Context context) {
__TRACE
  Point<image_region_dimensions> p0 = mImageDescriptor.origin();
  Point <image_region_dimensions> p1 = mImageDescriptor.upperBound() - Point<image_region_dimensions>::ONES();
  Rect<image_region_dimensions> imageBounds(p0, p1);
  domain = Domain(imageBounds);
__TRACE
  indexSpace = mRuntime->create_index_space(context, domain);
  fields = imageFields(context);
__TRACE
  region = mRuntime->create_logical_region(context, indexSpace, fields);
__TRACE
  mRuntime->attach_name(region, "sourceImage");
__TRACE
  fieldID[0] = FID_FIELD_R;
  fieldID[1] = FID_FIELD_G;
  fieldID[2] = FID_FIELD_B;
  fieldID[3] = FID_FIELD_A;
  fieldID[4] = FID_FIELD_Z;
  fieldID[5] = FID_FIELD_USERDATA;
__TRACE
  // fill the region initially with ZEROES
__TRACE
  PixelField zero = 0;
__TRACE
  TaskArgument arg(&zero, sizeof(zero));
__TRACE
  FillLauncher fillLauncher(region, region, arg);
__TRACE
  fillLauncher.add_field(FID_FIELD_R);
  fillLauncher.add_field(FID_FIELD_G);
  fillLauncher.add_field(FID_FIELD_B);
  fillLauncher.add_field(FID_FIELD_A);
  fillLauncher.add_field(FID_FIELD_Z);
  fillLauncher.add_field(FID_FIELD_USERDATA);
__TRACE
  mRuntime->fill_fields(context, fillLauncher);
__TRACE
}


void ImageReduction::partitionImageByImageDescriptor(LogicalRegion image, Context ctx, Runtime* runtime, ImageDescriptor imageDescriptor) {
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
                                            LogicalPartition sourcePartition, Context ctx, Runtime* runtime, ImageDescriptor imageDescriptor) {
__TRACE
  mRenderImageColorSpace = imageDescriptor.simulationColorSpace;
  // Legion::Point<image_region_dimensions> *coloring = new Legion::Point<image_region_dimensions>[mSimulationKDTree->size()];
  // mSimulationKDTree->getColorMap(coloring);
__TRACE
  // create a logical region to hold the coloring and extent
  Point<image_region_dimensions> p0 = mImageDescriptor.origin();
  Point <image_region_dimensions> p1 = mImageDescriptor.upperBound() - Point<image_region_dimensions>::ONES();
  Rect<image_region_dimensions> imageBounds(p0, p1);
  IndexSpace coloringIndexSpace = mRuntime->create_index_space(ctx, mRenderImageColorSpace);
  FieldSpace coloringFields = mRuntime->create_field_space(ctx);
  mRuntime->attach_name(coloringFields, "render image coloring fields");

__TRACE
  // FieldAllocator coloringAllocator = mRuntime->create_field_allocator(ctx, coloringFields);
  // FieldID fidColor = coloringAllocator.allocate_field(sizeof(Point<image_region_dimensions>), FID_FIELD_COLOR);
  // assert(fidColor == FID_FIELD_COLOR);
  // FieldID fidExtent = coloringAllocator.allocate_field(sizeof(Rect<image_region_dimensions>), FID_FIELD_EXTENT);
  // assert(fidExtent == FID_FIELD_EXTENT);

#if 0
  IndexPartition renderImageIP; {
__TRACE
  LogicalRegion coloringExtentRegion = mRuntime->create_logical_region(ctx, coloringIndexSpace, coloringFields);

__TRACE
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

__TRACE
  const FieldAccessor<WRITE_DISCARD, Rect<image_region_dimensions>,
  image_region_dimensions, long long int,
  Realm::AffineAccessor<Rect<image_region_dimensions>, image_region_dimensions, long long int> >
  acc_extent(coloringPhysicalRegion, FID_FIELD_EXTENT);

  Rect<image_region_dimensions> rect = imageBounds;

__TRACE
  for(unsigned i = 0; i < mSimulationKDTree->size(); ++i) {
    rect.lo.z = rect.hi.z = i;
    acc_extent[coloring[i]] = rect;
    acc_color[coloring[i]] = coloring[i];
#if 1
std::cout << "coloring[" << i << "] extent " << rect << " color " << coloring[i]  << std::endl;
#endif
  }
  // partition the coloring region by field
  IndexPartition coloringIP = mRuntime->create_partition_by_field(ctx,
    coloringExtentRegion, coloringExtentRegion, FID_FIELD_COLOR,
    coloringIndexSpace, AUTO_GENERATE_ID, gMapperID);
  LogicalPartition coloringPartition = runtime->get_logical_partition(ctx, coloringExtentRegion, coloringIP);

  renderImageIP = mRuntime->create_partition_by_image_range(
    ctx, mSourceIndexSpace, coloringPartition, coloringExtentRegion,
    FID_FIELD_EXTENT, coloringIndexSpace, DISJOINT_COMPLETE_KIND, AUTO_GENERATE_ID,
    gMapperID);
  }

#else

  std::map<DomainPoint, Domain> domains;
  size_t count = 0;

  for(Domain::DomainPointIterator it(runtime->get_index_space_domain(ctx, coloringIndexSpace)); it; it++) {
    DomainPoint color(it.p);
    Rect<3> base = imageBounds;
    base.lo.z = base.hi.z = count++;
    domains[color] = Domain(base);
    // std::cout << count << " point: " << color << " domains[color]: " << domains[color] << std::endl;
  }

  IndexPartition renderImageIP = 
    mRuntime->create_partition_by_domain(ctx, mSourceIndexSpace, domains, coloringIndexSpace, true, DISJOINT_COMPLETE_KIND);

#endif

  mRenderImagePartition = runtime->get_logical_partition(ctx, mSourceImage, renderImageIP);
  mRuntime->attach_name(mRenderImagePartition, "render image partition");

  // Legion::Rect<3> parent_bound = runtime->get_index_space_domain(ctx, mSourceIndexSpace);
  // std::cout << "parent_bound " << parent_bound << std::endl;
  // for(Domain::DomainPointIterator it(runtime->get_index_space_domain(ctx, coloringIndexSpace)); it; it++) {
  //   DomainPoint color(it.p);
  //   IndexSpace subregion = runtime->get_index_subspace(ctx, mRenderImagePartition.get_index_partition(), color);
  //   Legion::Rect<3> subdomain_bounds = runtime->get_index_space_domain(ctx, subregion);
  //   std::cout << "color: " << color  << " subdomain_bounds " << subdomain_bounds << std::endl;
  // }
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
  return 100 + level * 2 + more; //TODO assign ids dynamically
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
                                  Runtime *runtime) {
__TRACE
  if(mSimulationKDTree != nullptr) {
    return;
  }

  Rect<image_region_dimensions> rect = imageDescriptor.simulationDomain;
  std::cout << "simulationDomain " << rect << std::endl;

  KDTreeValue* simulationElements = new KDTreeValue[rect.volume()];
  unsigned index = 0;
  Point<image_region_dimensions> p0 = Point<image_region_dimensions>::ZEROES();
  Rect<image_region_dimensions> zeroRect(p0, p0);

__TRACE
  for(Domain::DomainPointIterator it(imageDescriptor.simulationDomain); it; it++) {
    DomainPoint color(it.p);
    IndexSpace subregion = runtime->get_index_subspace(ctx,
      imageDescriptor.simulationLogicalPartition.get_index_partition(), color);
    Domain subdomain = runtime->get_index_space_domain(ctx, subregion);
    // std::cout << "subdomain " << Legion::Rect<3>(subdomain) << std::endl;
    Legion::Rect<image_region_dimensions> simulationRect(color, color);
    KDTreeValue simulationValue;
    simulationValue.extent = simulationRect;
    simulationValue.color = color;
    simulationValue.extent2 = zeroRect;
    simulationValue.world_space_bounds = subdomain;
    simulationElements[index] = simulationValue;
    index++;
  }

__TRACE
  mSimulationKDTree = new KDTree<image_region_dimensions, long long int>(simulationElements, rect.volume());
  delete [] simulationElements;

  Rect<image_region_dimensions>* simulationExtents = new Rect<image_region_dimensions>[rect.volume()];
  mSimulationKDTree->getExtent(simulationExtents);
  KDTreeValue* imageElements = new KDTreeValue[rect.volume()];

__TRACE
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

__TRACE
  delete [] simulationExtents;
  mImageKDTree = new KDTree<image_region_dimensions, long long int>(imageElements, rect.volume());
  delete [] imageElements;
__TRACE
}


void ImageReduction::initial_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime) {

#ifdef TRACE_TASKS
  std::cout << describe_task(task) << std::endl;
#endif
__TRACE
  ImageDescriptor imageDescriptor = *((ImageDescriptor*)task->args);
  createProjectionFunctors(runtime, imageDescriptor.numImageLayers);
  if(imageDescriptor.hasPartition) {
__TRACE
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



void ImageReduction::initializeRenderNodes(Runtime* runtime,
                                           Context context,
                                           unsigned taskID,
                                           char* args,
                                           int totalArgLen) {
  ArgumentMap argMap;
  char *argsBuffer = new char[totalArgLen];
  memcpy(argsBuffer, args, totalArgLen);

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
    partition = mCompositeImagePartition;
    region = mSourceImage;
  }

  IndexTaskLauncher launcher(taskID, domain,
                             TaskArgument(argsBuffer, totalArgLen), argMap, Predicate::TRUE_PRED,
                             false, gMapperID);
  RegionRequirement req(partition, 0, READ_ONLY, EXCLUSIVE, region, gMapperID);
  FutureMap futures = runtime->execute_index_space(context, launcher);
  futures.wait_all_results();
  delete [] argsBuffer;
}


void ImageReduction::initializeNodes(Runtime* runtime, Context context) {
__TRACE
  unsigned taskID = mInitialTaskID;
  ArgumentMap argMap;
  int totalArgLen = sizeof(mImageDescriptor);
  char *argsBuffer = new char[totalArgLen];
  memcpy(argsBuffer, &mImageDescriptor, sizeof(mImageDescriptor));

__TRACE
  // if imageDescriptor has a partition launch over the partition
  // otherwise launch over the image compositeImageDomain
  Domain domain;
  LogicalPartition partition;
  LogicalRegion region;
  if(mImageDescriptor.hasPartition) {
__TRACE
    domain = mImageDescriptor.simulationDomain;
    partition = mImageDescriptor.simulationLogicalPartition;
    region = mImageDescriptor.simulationLogicalRegion;
  } else {
    domain = mCompositeImageDomain;
    partition = mCompositeImagePartition;
    region = mSourceImage;
  }

  // IndexSpace _indexSpace = region.get_index_space();
  // Domain _domain = runtime->get_index_space_domain(context, _indexSpace);
  // Legion::Rect<3> bounds = _domain;
  // std::cout << "simulationLogicalRegion bounds\n"
  //           << bounds.lo[0] << " " << bounds.lo[1] << " " << bounds.lo[2] << "\n"
  //           << bounds.hi[0] << " " << bounds.hi[1] << " " << bounds.hi[2] << "\n";

__TRACE
  IndexTaskLauncher launcher(taskID, domain,
                             TaskArgument(argsBuffer, totalArgLen), argMap, Predicate::TRUE_PRED,
                             false, gMapperID);
  RegionRequirement req(partition, 0, READ_ONLY, EXCLUSIVE, region, gMapperID);
  if(mImageDescriptor.hasPartition) {
    for(int i = 0; i < mImageDescriptor.numPFields; ++i) {
      req.add_field(mImageDescriptor.pFields[i]);
    }
  } else {
    addImageFieldsToRequirement(req);
  }
  launcher.add_region_requirement(req);
__TRACE
  FutureMap futures = runtime->execute_index_space(context, launcher);
  futures.wait_all_results();
  delete [] argsBuffer;
__TRACE
  if(mImageDescriptor.hasPartition) {
__TRACE
    buildKDTrees(mImageDescriptor, context, runtime);
__TRACE
  }
}


Legion::FutureMap
ImageReduction::launch_task_composite_domain(
                                                  unsigned taskID,
                                                  Runtime* runtime,
                                                  Context context,
                                                  void* argsBuffer,
                                                  int totalArgLen,
                                             bool blocking) {
  ArgumentMap argMap;
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
    partition = mCompositeImagePartition;
    region = mSourceImage;
  }

  IndexTaskLauncher launcher(taskID, domain,
                             TaskArgument(argsBuffer, totalArgLen), argMap, Predicate::TRUE_PRED,
                             false, gMapperID);
  RegionRequirement req(partition, 0, READ_WRITE, EXCLUSIVE, region, gMapperID);
  if(mImageDescriptor.hasPartition) {
    for(int i = 0; i < mImageDescriptor.numPFields; ++i) {
      req.add_field(mImageDescriptor.pFields[i]);
    }
  } else {
    addImageFieldsToRequirement(req);
  }
  launcher.add_region_requirement(req);
  FutureMap futures = runtime->execute_index_space(context, launcher);
  if(blocking) futures.wait_all_results();
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

template<int N>
static Legion::Point<N> cal_center(const Legion::Rect<N>& bbox) {
  Legion::Point<N> center;
  for (int i = 0; i < N; ++i) {
    center[i] = 0.5 * (bbox.lo[i] + bbox.hi[i]);
  }
  return center;
}

template<int N>
static double cal_distance2(const Legion::Point<N>& a, const Legion::Point<N>& b) {
  double dist = 0.;
  for (int i = 0; i < N; ++i) {
    const double d = (a[i] - b[i]);
    dist += d*d;
  }
  return dist;
}

template<int N>
static void print(const Legion::Point<N>& p) {
  std::cout << "(";
  for (int i = 0; i < N; ++i) {
    std::cout << p[i] << " ";
  }
  std::cout << ")";
}

bool ImageReduction::flipRegions(PhysicalRegion fragment0,
                                 PhysicalRegion fragment1,
                                 bool cameraIsOrthographic,
                                 float cameraData[image_region_dimensions]) 
{
  if(mSimulationKDTree == nullptr) return false;
  if(cameraData == nullptr) return false;

  KDNode<image_region_dimensions, long long int>* node0 = findFragmentInKDTree(fragment0);
  KDNode<image_region_dimensions, long long int>* node1 = findFragmentInKDTree(fragment1);

  std::cout 
          << std::endl << "frag 0 bbox " 
          << node0->mValue.world_space_bounds.lo[0] << " " << node0->mValue.world_space_bounds.lo[1] << " " << node0->mValue.world_space_bounds.lo[2] << ", "
          << node0->mValue.world_space_bounds.hi[0] << " " << node0->mValue.world_space_bounds.hi[1] << " " << node0->mValue.world_space_bounds.hi[2] << " "
          << std::endl << "frag 1 bbox " 
          << node1->mValue.world_space_bounds.lo[0] << " " << node1->mValue.world_space_bounds.lo[1] << " " << node1->mValue.world_space_bounds.lo[2] << ", "
          << node1->mValue.world_space_bounds.hi[0] << " " << node1->mValue.world_space_bounds.hi[1] << " " << node1->mValue.world_space_bounds.hi[2] << " "
          << std::endl;

  if (cameraIsOrthographic) {
    unsigned axis0 = node0->mLevel % image_region_dimensions; // image_region_dimensions === 3
    unsigned axis1 = node1->mLevel % image_region_dimensions;
    float splittingPlaneNormal[image_region_dimensions] = { 0 };
    if (axis0 == axis1) {
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

    // Legion::Domain domain0 = fragment0.get_bounds<image_region_dimensions, long long>();
    // Legion::Domain domain1 = fragment1.get_bounds<image_region_dimensions, long long>();
    // std::cout << "splittingPlaneNormal " 
    //         << splittingPlaneNormal[0] << " "
    //         << splittingPlaneNormal[1] << " "
    //         << splittingPlaneNormal[2] << std::endl;
    // std::cout << "cameraDirection " 
    //           << cameraDirection[0] << " "
    //           << cameraDirection[1] << " "
    //           << cameraDirection[2] << std::endl;

    float dot = 0;
    for(unsigned i = 0; i < image_region_dimensions; ++i) {
      dot += splittingPlaneNormal[i] * cameraData[i]; // what is the equivalent for perspective projection ???
    }
    return dot < 0;
  }
  else {
    Legion::Point<image_region_dimensions> center0, center1, camPos;
    center0 = cal_center(node0->mValue.world_space_bounds);
    center1 = cal_center(node1->mValue.world_space_bounds);
    for (int i = 0; i < image_region_dimensions; ++i) {
      camPos[i] = cameraData[i];
    }
    std::cout << "center0 "; print(center0); std::cout << std::endl;
    std::cout << "center1 "; print(center1); std::cout << std::endl;
    std::cout << "camPos  "; print(camPos); std::cout << std::endl;

    return cal_distance2(center0, camPos) > cal_distance2(center1, camPos);
  }
}


void ImageReduction::composite_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx,
                                    Runtime *runtime) {
#ifdef TRACE_TASKS
  std::cout << describe_task(task) << std::endl;
#endif

#if NULL_COMPOSITE_TASKS
  return; // performance testing
#endif

  CompositeArguments args = ((CompositeArguments*)task->args)[0];
  PhysicalRegion fragment0 = regions[0];
  PhysicalRegion fragment1 = regions[1];

  Legion::Domain domain0 = fragment0.get_bounds<image_region_dimensions, long long>();
  Legion::Domain domain1 = fragment1.get_bounds<image_region_dimensions, long long>();
  int Z0 = domain0.lo()[2];
  int Z1 = domain1.lo()[2];

  ImageReductionComposite::CompositeFunction* compositeFunction;
  compositeFunction = ImageReductionComposite::compositeFunctionPointer(args.depthFunction, args.blendFunctionSource, args.blendFunctionDestination, args.blendEquation);

  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r0(fragment0, FID_FIELD_R);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g0(fragment0, FID_FIELD_G);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b0(fragment0, FID_FIELD_B);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a0(fragment0, FID_FIELD_A);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z0(fragment0, FID_FIELD_Z);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata0(fragment0, FID_FIELD_USERDATA);

  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r1(fragment1, FID_FIELD_R);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g1(fragment1, FID_FIELD_G);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b1(fragment1, FID_FIELD_B);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a1(fragment1, FID_FIELD_A);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z1(fragment1, FID_FIELD_Z);
  const FieldAccessor<READ_WRITE, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata1(fragment1, FID_FIELD_USERDATA);

  bool flip = flipRegions(fragment0, fragment1, args.cameraIsOrthographic, args.cameraData);
#if 0
  std::cout << __FUNCTION__ << " Z0 " << Z0 << " Z1 " << Z1 << " (r0,g0,b0) << " << r0[0] << " " << g0[0] << " " << b0[0] << " (r1,g1,b1) " << r1[0] << " " << g1[0] << " " << b1[0] << std::endl;
#endif
  compositeFunction(r0, g0, b0, a0, z0, userdata0, r1, g1, b1, a1, z1, userdata1,
                    args.imageDescriptor.width, args.imageDescriptor.height, Z0, Z1, flip);


}





FutureMap ImageReduction::launchTreeReduction(ImageDescriptor imageDescriptor, int treeLevel,
                                              GLenum depthFunc, GLenum blendFuncSource,
                                              GLenum blendFuncDestination, GLenum blendEquation,
                                              int compositeTaskID, LogicalPartition sourcePartition,
                                              LogicalRegion image,
                                              Runtime* runtime, Context context,
                                              int maxTreeLevel,
                                              bool cameraIsOrthographic, float cameraData[image_region_dimensions])
{
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
  args.cameraIsOrthographic = cameraIsOrthographic;
  memcpy(args.cameraData, cameraData, sizeof(args.cameraData));
  IndexTaskLauncher treeCompositeLauncher(compositeTaskID, launchDomain,
                                          TaskArgument(&args, sizeof(args)), argMap, Predicate::TRUE_PRED, false,
                                          gMapperID);

  RegionRequirement req0(sourcePartition, functor0->id(),
                         READ_WRITE, EXCLUSIVE, image, gMapperID);
  addImageFieldsToRequirement(req0);
  treeCompositeLauncher.add_region_requirement(req0);

  RegionRequirement req1(sourcePartition, functor1->id(),
                         READ_WRITE, EXCLUSIVE, image, gMapperID);
  addImageFieldsToRequirement(req1);
  treeCompositeLauncher.add_region_requirement(req1);

  FutureMap futures = runtime->execute_index_space(context, treeCompositeLauncher);

  if(treeLevel > 1) {
    futures = launchTreeReduction(imageDescriptor, treeLevel - 1, depthFunc,
                                  blendFuncSource, blendFuncDestination, blendEquation, compositeTaskID,
                                  sourcePartition, image, runtime, context, maxTreeLevel, 
                                  cameraIsOrthographic, cameraData);
  }

  return futures;
}

FutureMap ImageReduction::reduceImagesOrthographic(Context context, float cameraDirection[]) {
  int maxTreeLevel = numTreeLevels(mImageDescriptor);
  if(maxTreeLevel > 0) {
    return launchTreeReduction(mImageDescriptor, maxTreeLevel, mDepthFunction,
                               mGlBlendFunctionSource, mGlBlendFunctionDestination, mGlBlendEquation,
                               mCompositeTaskID, mCompositeImagePartition, mSourceImage, mRuntime,
                               context, maxTreeLevel, true, cameraDirection);
  } else {
    return FutureMap();
  }
}

FutureMap ImageReduction::reduceImagesPerspective(Context context, float cameraLocation[]) {
  int maxTreeLevel = numTreeLevels(mImageDescriptor);
  if(maxTreeLevel > 0) {
    return launchTreeReduction(mImageDescriptor, maxTreeLevel, mDepthFunction,
                               mGlBlendFunctionSource, mGlBlendFunctionDestination, mGlBlendEquation,
                               mCompositeTaskID, mCompositeImagePartition, mSourceImage, mRuntime,
                               context, maxTreeLevel, false, cameraLocation);
  } else {
    return FutureMap();
  }
}


void ImageReduction::display_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime) {

#ifdef TRACE_TASKS
  std::cout << describe_task(task) << std::endl;
#endif
  DisplayArguments args = ((DisplayArguments*)task->args)[0];
  char fileName[1024];
  sprintf(fileName, "display.%d.tga", args.t);
  string outputFileName = string(fileName);
  PhysicalRegion displayPlane = regions[0];

  const FieldAccessor<READ_ONLY, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > r(displayPlane, FID_FIELD_R);
  const FieldAccessor<READ_ONLY, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > g(displayPlane, FID_FIELD_G);
  const FieldAccessor<READ_ONLY, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > b(displayPlane, FID_FIELD_B);
  const FieldAccessor<READ_ONLY, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > a(displayPlane, FID_FIELD_A);
  const FieldAccessor<READ_ONLY, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > z(displayPlane, FID_FIELD_Z);
  const FieldAccessor<READ_ONLY, ImageReduction::PixelField, image_region_dimensions, coord_t, Realm::AffineAccessor<ImageReduction::PixelField, image_region_dimensions, coord_t> > userdata(displayPlane, FID_FIELD_USERDATA);

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
      GLubyte b_ = float(b[x][y][0]) * 255;
      fputc(b_, f); /* write blue */
      GLubyte g_ = float(g[x][y][0]) * 255;
      fputc(g_, f); /* write green */
      GLubyte r_ = float(r[x][y][0]) * 255;
      fputc(r_, f);   /* write red */
    }
  }
  fclose(f);
  std::cout << "wrote image " << outputFileName << std::endl;

}



Future ImageReduction::display(int t, Context context) {
  DisplayArguments args = { mImageDescriptor, t };
  TaskLauncher taskLauncher(mDisplayTaskID, TaskArgument(&args, sizeof(args)),
                            Predicate::TRUE_PRED, gMapperID);
  DomainPoint origin = DomainPoint(Point<image_region_dimensions>::ZEROES());
  LogicalRegion displayPlane = mRuntime->get_logical_subregion_by_color(mCompositeImagePartition, origin);
  RegionRequirement req(displayPlane, READ_ONLY, EXCLUSIVE,
                        mSourceImage, gMapperID);
  addImageFieldsToRequirement(req);
  taskLauncher.add_region_requirement(req);
  Future displayFuture = mRuntime->execute_task(context, taskLauncher);
  return displayFuture;
}



}
}
