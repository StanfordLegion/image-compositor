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
See "Blending todo" for notes on how to implement the other reductions,

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
These use cases differ in how this logical partition is defined,  in whether the rendering is done in C++ or in Regent, and whether they use the ImageReductionMapper..
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
##### cxx_initialize
This entry point is called once when the program starts up.
It creates an ImageCompositor that respects the fluidPartition.
It returns a struct (type RegionPartition) that contains several legion objects related to the source image region.
```
// this entry point is called once from the main task
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
##### cxx_reduce


### Regent application with rendering in Regent

## Blending todo

