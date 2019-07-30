# image-compositor

## Introduction
This is an image compositor framework for Legion.  It allows Legion applications to render data in situ and in parallel and reduce the resulting plurality of images down to a single image for display.

![](./doc/isav_workshop_sc17_presentation.mp4.gif)

Responsibility for rendering images is left up to the application.
The application can render data from either Regent or C++.
The framework will composite those rendered images to produce a reduced result.

## Requirements
The framework supports the standard OpenGL Depth and Blend compositing functions in a distributed environment.
These are sufficient for most scientific visualization purposes such as surface and volume rendering.
The framework supports compositing functions of the following algebraic types:
associative-commutative, associative-noncommutative, nonassociative-noncommutative.
Surface rendering uses the associative-commutative depth functions, while volume rendering uses the associative-noncommutative blend functions.

Note: following the philosophy of "don't build it until there is a customer" the framework has only implemented the associate-commutative reductions.
See "Blending todo" and "Nonassociative todo" below for notes on how to implement the other reductions.

## Code organization

```
image-compositor
├── examples
│   └── visualization
├── include
├── src
└── test
    ├── visualization_test_1
    ├── visualization_test_2
    └── visualization_test_3
```

### framework
The framework is contained in src/ and include/.
It is built with cmake.
To build the framework do this:
```
cd image-compositor

cmake .
or
cmake -DCMAKE_BUILD_TYPE=Debug .

make
```
#### mapper
The framework contains a customer mapper in src/.
During application startup the mapper is provided with the name of the Render function.
The mapper will track the address spaces where the Render function executes, and will ensure that subsequent compositing functions are mapped to the same address spaces.
This gives the application full control over where the data resides and where rendering takes place.

### tests
Tests exist for associative-commutative reductions and surface rendering.
Tests exercise the full range of OpenGL depth and blend functions.
Since many of the blend functions are non-commutative the values computed in these tests are incorrect.
To build the tests do this:
```
cd test
make
```

#### running tests
Run the tests using the run* bash scripts in the test/visualization_test_* directories.

### examples
The visualization example is a basic C++ example that does not define a simulation domain.
We need to develop a Regent example that defines a simulation domain.
To make the visualization example do this:
```
cd examples/visualization
make
```

#### running examples
Run the examples using the run* bash scripts in the examples/* directories.

## Use cases
The framework was originally developed in C++ for use with Legion applications.
It can also be used with Regent programs.
The framework uses a logical partition as a source for the image reduction.
These use cases differ in how this logical partition is defined,  in whether the rendering is done in C++ or in Regent, and whether they use the ImageReductionMapper.
These use cases are described here.

### Tests and visualization example
In this case the logical partition is created for a source image according to an image descriptor.
The index space of the partition is required to be 3D.
Rendering and reduction are performed in C++.
This case does not use the ImageReductionMapper.

### C++ Legion application
In this case the application has defined a simulation domain.
It provides a logical partition of this domain to the ImageCompositor constructor.
It defines the name of the Render task by calling ImageReductionMapper::registerRenderTaskName().
The application index launches the Render tasks using ImageReductionProjectFunctors that mediate between the simulation domain and the source image domain.
When the Render tasks execute the mapper records the address spaces where the tasks are mapped.
The application invokes
```
compositor->reduce_associative_commutative()
```
which index launches the image reduction operations in the address spaces that were recorded previously by the mapper.

### Regent application with rendering in C++/OpenGL
This is the most common use case for applications that use OpenGL or similar C++ graphics APIs.

#### Soleil-x
Soleil-x is  combined fluid-particle-radiation simulation written in Regent.
It is located here:
https://github.com/stanfordhpccenter/soleil-x/tree/feat/viz/4

The relevant parts of the Soleil-x source code are in render.h and render.cpp.
These export symbols cxx_preinitialize, cxx_initialize, cxx_render and cxx_reduce.

##### cxx_preinitialize
This is called from the mapper before the runtime starts and performs certain initializations.
The soleil_mapper ends with this code:
```
static MapperID imageReductionMapperID;
class ImageReductionMapper : public DefaultMapper {
public:
  ImageReductionMapper(MapperRuntime* rt, Machine machine, Processor local);
};

static void create_mappers(Machine machine,
                           Runtime* rt,
                           const std::set<Processor>& local_procs) {
  for (Processor proc : local_procs) {
    rt->replace_default_mapper(new SoleilMapper(rt, machine, proc), proc);
    ImageReductionMapper* irMapper =
      new ImageReductionMapper(rt->get_mapper_runtime(), machine, proc);
    rt->add_mapper(imageReductionMapperID, (Mapping::Mapper*)irMapper, proc);
  }
}

#ifdef __cplusplus
extern "C" {
#endif

  void cxx_preinitialize(MapperID);
  
#ifdef __cplusplus
}
#endif

void register_mappers() {
  imageReductionMapperID = Legion::Runtime::generate_static_mapper_id();
  cxx_preinitialize(imageReductionMapperID);
  Runtime::add_registration_callback(create_mappers);
}
```

The register_mappers() function is invoked from the Regent program like this:
```
local MAPPER = terralib.includec("soleil_mapper.h")

regentlib.saveobj(main, "soleil.o", "object", MAPPER.register_mappers)
```

##### cxx_initialize
This entry point is called once when the program starts up.
It creates an ImageCompositor that respects the fluidPartition.
It returns a struct (type RegionPartition) that contains several legion objects related to the source image region.
```
RegionPartition cxx_initialize(
                                    legion_runtime_t runtime_,
                                    legion_context_t ctx_,
                                    legion_mapper_id_t sampleId,
                                    legion_logical_partition_t fluidPartition_
                                    )
  {
    Runtime *runtime = CObjectWrapper::unwrap(runtime_);
    Context ctx = CObjectWrapper::unwrap(ctx_)->context();
    LogicalPartition fluidPartition = CObjectWrapper::unwrap(fluidPartition_);

    // Initialize an image compositor, or reuse an initialized one
    
    Visualization::ImageDescriptor imageDescriptor = { gImageWidth, gImageHeight, 1 };
    
    if(gImageCompositors.find(sampleId) == gImageCompositors.end()) {
      gImageCompositors[sampleId] = new Visualization::ImageReduction(fluidPartition, imageDescriptor, ctx, runtime, gImageReductionMapperID);
      ImageReductionMapper::registerRenderTaskName("render_task");
    }
    
    Visualization::ImageReduction* compositor = gImageCompositors[sampleId];
    RegionPartition result;
    result.indexSpace = CObjectWrapper::wrap(compositor->sourceIndexSpace());
    result.imageX = CObjectWrapper::wrap(compositor->sourceImage());
    result.colorSpace = CObjectWrapper::wrap(compositor->everywhereColorSpace());
    result.p_Image = CObjectWrapper::wrap(compositor->everywherePartition());
    compositor->sourceImageFields(ctx,    result.imageFields);
    return result;
  }
```

##### cxx_render
This entry point is called each time the application wants to perform a parallel rendering operation.
The application passes the simulation data in.
The entry point performs an index launch of the Render task and uses ImageReductionProjectionFunctor objects to map between the simulation domain and the source image.
```
  void cxx_render(legion_runtime_t runtime_,
                  legion_context_t ctx_,
                  legion_mapper_id_t sampleId,
                  legion_physical_region_t* fluid_,
                  legion_field_id_t fluidFields[],
                  int numFluidFields,
                  legion_physical_region_t* particles_,
                  legion_field_id_t particlesFields[],
                  int numParticlesFields,
                  legion_physical_region_t* image_,
                  legion_field_id_t imageFields[],
                  int numImageFields,
                  legion_logical_partition_t fluidPartition_,
                  legion_logical_partition_t particlesPartition_,
                  int numParticlesToDraw,
                  int isosurfaceField,
                  double isosurfaceValue,
                  double isosurfaceScale[2],
                  legion_physical_region_t* particlesToDraw_,
                  FieldData lowerBound[3],
                  FieldData upperBound[3]
                  )
  {
    std::cout << __FUNCTION__ << " pid " << getpid() << std::endl;
    static bool firstTime = true;
    
    // Unwrap objects
    
    Runtime *runtime = CObjectWrapper::unwrap(runtime_);
    Context ctx = CObjectWrapper::unwrap(ctx_)->context();
    PhysicalRegion* fluid = CObjectWrapper::unwrap(fluid_[0]);
    LogicalPartition fluidPartition = CObjectWrapper::unwrap(fluidPartition_);
    PhysicalRegion* particles = CObjectWrapper::unwrap(particles_[0]);
    LogicalPartition particlesPartition = CObjectWrapper::unwrap(particlesPartition_);
    PhysicalRegion* particlesToDraw = CObjectWrapper::unwrap(particlesToDraw_[0]);
    PhysicalRegion* image = CObjectWrapper::unwrap(image_[0]);

    // Create projection functors
    
    Visualization::ImageReduction* compositor = gImageCompositors[sampleId];
    if(firstTime) {
      ImageReductionProjectionFunctor* functor0 = new ImageReductionProjectionFunctor(compositor->everywhereDomain(), fluidPartition);
      runtime->register_projection_functor(1, functor0);
      ImageReductionProjectionFunctor* functor1 = new ImageReductionProjectionFunctor(compositor->everywhereDomain(), particlesPartition);
      runtime->register_projection_functor(2, functor1);
      ImageReductionProjectionFunctor* functor2 = new ImageReductionProjectionFunctor(compositor->everywhereDomain(), compositor->everywherePartition());
      runtime->register_projection_functor(3, functor2);
    }
    
    // Construct arguments to render task

    ArgumentMap argMap;
    ImageDescriptor imageDescriptor = compositor->imageDescriptor();
    size_t argSize = sizeof(imageDescriptor) + 6 * sizeof(FieldData) + sizeof(int) + sizeof(VisualizationField) + sizeof(double) + 2 * sizeof(double) +
      numParticlesToDraw * sizeof(long int);
    char args[argSize];
    char* argsPtr = args;
    memcpy(argsPtr, &imageDescriptor, sizeof(imageDescriptor));
    argsPtr += sizeof(imageDescriptor);
    memcpy(argsPtr, lowerBound, 3 * sizeof(FieldData));
    argsPtr += 3 * sizeof(FieldData);
    memcpy(argsPtr, upperBound, 3 * sizeof(FieldData));
    argsPtr += 3 * sizeof(FieldData);
    memcpy(argsPtr, &numParticlesToDraw, sizeof(int));
    argsPtr += sizeof(int);
    memcpy(argsPtr, &isosurfaceField, sizeof(VisualizationField));
    argsPtr += sizeof(VisualizationField);
    memcpy(argsPtr, &isosurfaceValue, sizeof(double));
    argsPtr += sizeof(double);
    memcpy(argsPtr, isosurfaceScale, 2 * sizeof(double));
    argsPtr += 2 * sizeof(double);
    
    // Copy particlesToDraw as a task argument
    
    std::vector<legion_field_id_t> particlesToDrawFields;
    particlesToDraw->get_fields(particlesToDrawFields);
    AccessorRO<long int, 1> particlesToDrawId(*particlesToDraw, particlesToDrawFields[0]);
 
    long int* longArgsPtr = (long int*)argsPtr;
    for(int i = 0; i < numParticlesToDraw; ++i) {
      Point<1> p = i;
      longArgsPtr[i] = particlesToDrawId[p];
    }

    // Setup the render task launch with region requirements
    
    IndexTaskLauncher renderLauncher(gRenderTaskID, compositor->everywhereDomain(), TaskArgument(args, argSize), argMap, Predicate::TRUE_PRED, false, gImageReductionMapperID);
    
    RegionRequirement req0(fluidPartition, 1, READ_ONLY, EXCLUSIVE, fluid->get_logical_region(), gImageReductionMapperID);
    for(int i = 0; i < numFluidFields; ++i) req0.add_field(fluidFields[i]);
    renderLauncher.add_region_requirement(req0);
    
    RegionRequirement req1(particlesPartition, 2, READ_ONLY, EXCLUSIVE, particles->get_logical_region(), gImageReductionMapperID);
    for(int i = 0; i < numParticlesFields; ++i) {
      req1.add_field(particlesFields[i]);
    }
    renderLauncher.add_region_requirement(req1);
    
    RegionRequirement req2(compositor->everywherePartition(), 3, WRITE_DISCARD, EXCLUSIVE, image->get_logical_region(), gImageReductionMapperID);
    req2.add_field(Visualization::ImageReduction::FID_FIELD_R);
    req2.add_field(Visualization::ImageReduction::FID_FIELD_G);
    req2.add_field(Visualization::ImageReduction::FID_FIELD_B);
    req2.add_field(Visualization::ImageReduction::FID_FIELD_A);
    req2.add_field(Visualization::ImageReduction::FID_FIELD_Z);
    renderLauncher.add_region_requirement(req2);
    
    // Clear the mapping history so render_task will create it anew
    
    ImageReductionMapper::clearPlacement(fluidPartition);
    
    // Launch the render task
    
    FutureMap futures = runtime->execute_index_space(ctx, renderLauncher);
    futures.wait_all_results();
    firstTime = false;
  }
```

##### cxx_reduce
This entry point is called immediately after cxx_render.
It invokes the compositor to perform the image reduction, and then uses a singleton task to save the image to disk.
```
  void cxx_reduce(legion_runtime_t runtime_,
                  legion_context_t ctx_,
                  legion_mapper_id_t sampleId,
                  const char* outDir
                  )
  {
#if !USE_COMPOSITOR
    return;
#endif
    std::cout << __FUNCTION__ << std::endl;
    
    Runtime *runtime = CObjectWrapper::unwrap(runtime_);
    Context ctx = CObjectWrapper::unwrap(ctx_)->context();
    
    Visualization::ImageReduction* compositor = gImageCompositors[sampleId];
    compositor->set_depth_func(GL_LESS);
    FutureMap futures = compositor->reduce_associative_commutative(ctx);
    futures.wait_all_results();
    
    // save the image
    ImageDescriptor imageDescriptor = compositor->imageDescriptor();
    size_t argLen = sizeof(ImageDescriptor) + strlen(outDir) + 1;
    char args[argLen] = { 0 };
    memcpy(args, &imageDescriptor, sizeof(imageDescriptor));
    strcpy(args + sizeof(imageDescriptor), outDir);
    TaskLauncher saveImageLauncher(gSaveImageTaskID, TaskArgument(args, argLen), Predicate::TRUE_PRED, gImageReductionMapperID);
    DomainPoint slice0 = Point<3>::ZEROES();
    LogicalRegion imageSlice0 = runtime->get_logical_subregion_by_color(compositor->everywherePartition(), slice0);
    RegionRequirement req(imageSlice0, READ_ONLY, EXCLUSIVE, compositor->sourceImage());
    saveImageLauncher.add_region_requirement(req);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_R);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_G);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_B);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_A);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_Z);
    Future result = runtime->execute_task(ctx, saveImageLauncher);
    result.get_result<int>();
  }
```

### Regent application with rendering in Regent
This is similar to the previous case, but without using cxx_render().
The application calls cxx_preinitialize and cxx_initialize as before.
Instead of cxx_render() the application index launches a render task in Regent.
The render task writes to the source image region created by the image compositor.
After launching the render task the application calls cxx_reduce.

## Blending todo
These notes describe how to implement ordered blending as is required for volume rendering.

We assume the render tasks are launched over all elements of a partition P.
At application startup a vector of coordinates is computed where each coordinate is the center coordinate of an element of P.
Before the application calls cxx_reduce it 
applies the current view transform to each element of this vector, and sorts the transformed result by the Z coordinate.
At the same time it sorts by Z it maintains a permutation vector that indicates 
the sorted position of each element.
After sorting this permutation vector gives the blending order of the elements of P.

The image reduction framework needs to blend images in order according to their line-of-sight from the camera.
This order is defined by the permutation vector computed above.
The reduce_associative_noncommutative() function will differ from its commutative counterpart in that it will use a custom projection functor that reads this permutation vector.
When the projection functor is constructed it is given the base address of the permutation vector.
The project() function takes in an index space coordinate i and returns a subregion of P that corresponds to the ith element of the permutation vector.

## Nonassociative todo
Associative reductions can be implemented in a tree.
Nonassociative reductions must be implemented in a serial chain.
Nonocommutative reductions subsume commutative reductions 
so it is suffiient to implement only the noncommutative form.

These can be implemented when there is a need.
