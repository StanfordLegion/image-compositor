# image-compositor

## Introduction
This is an image compositor framework for Legion.  It allows Legion applications to render data in situ and in parallel and reduce the resulting plurality of images down to a single image for display.

![](./doc/isav_workshop_sc17_presentation.mp4.gif)

Responsibility for rendering images is left up to the application.
The application can render data from either Regent or C++.
The framework will composite those rendered images to produce a reduced result.

## Prior reading
To understand this framework it is necesary to be familiar with the following papers:

[A Sorting Classification of Parallel Rendering.  Molnar, Cox, Ellsworth and Fuchs (1994).](http://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/molnar94_sorting.pdf)

[Compositing Digital Images.  Porter, Duff (1984).](http://www.cs.cornell.edu/courses/cs465/2003fa/readings/cuonly/p253-porter.pdf)


## Requirements
The framework supports the standard OpenGL Depth and Blend compositing functions in a distributed environment.
These are sufficient for most scientific visualization purposes such as surface and volume rendering.
The framework supports compositing functions of the following algebraic types:
associative-commutative, associative-noncommutative, nonassociative-noncommutative.
Surface rendering uses the associative-commutative depth functions, while volume rendering uses the associative-noncommutative blend functions.


## Code organization

```
image-compositor
├── examples
│   ├── visualization_1
|   └── visualization_2
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
There are two example programs.
visualization_1 is a basic C++ example that does not define a simulation domain.
visualization_2 is a more extensive example that tests blending in the presence of a simulation domain.
You should follow this example to use the framework with an existing simulation.
To make the visualization examples do this:
```
cd examples/visualization[1|2]
make
```

#### running examples
Run the examples using the run* bash scripts in the examples/* directories.

## Use cases
The framework was originally developed in C++ for use with Legion applications.
It can also be used with Regent programs.
The framework uses a logical partition as a source for the image reduction.
These use cases differ in how this logical partition is defined, and whether the rendering is done in C++ or in Regent.
These use cases are described here.

### Tests and visualization_1 example
In this case the logical partition is created for a source image according to an image descriptor.
The index space of the partition is required to be 3D.
Rendering and reduction are performed in C++.

### C++ Legion application
In this case the application has defined a simulation domain.
It provides a logical partition of this domain to the ImageCompositor constructor.

The application invokes
```
compositor->reduceImages(ctx, cameraDirection)
```
which index launches the image reduction operations in the address spaces that were recorded previously by the mapper.
ctx is a Legion context.
cameraDirection is the difference between the camera lookAt point and the camera from point.
In other words it is the viewing direction of the camera.

### Regent application with rendering in C++/OpenGL
This is the most common use case for applications that use OpenGL or similar C++ graphics APIs.

#### visualization_2
This example shows how to use the framework with an existing simulation written in Regent.
The relevant parts of the C++ source code are contained in render.h, render.cc and renderCube.cc.
These export symbols cxx_preinitialize, cxx_initialize, cxx_render, cxx_reduce and cxx_saveImage.
In the following program r and p represent the simulation region and partition.

```
import "regent"
c = regentlib.c


-------------------------------------------------------------------------------
-- Visualization
-------------------------------------------------------------------------------

local root_dir = arg[0]:match(".*/") or "./"
assert(os.getenv('LG_RT_DIR'), "LG_RT_DIR should be set!")
local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
local legion_dir = runtime_dir .. "legion/"
local mapper_dir = runtime_dir .. "mappers/"
local realm_dir = runtime_dir .. "realm/"

render = terralib.includec("render.h",
{"-I", root_dir,
"-I", runtime_dir,
"-I", mapper_dir,
"-I", legion_dir,
"-I", realm_dir,
})

struct Image_columns {
  R : float,
  G : float,
  B : float,
  A : float,
  Z : float,
  U : float
}




terra configureCamera(angle : float)
  var camera : render.Camera
  if angle < -180 then angle = angle + 360 end
  if angle > 180 then angle = angle - 360 end
  camera.up[0] = 0
  camera.up[1] = 1
  camera.up[2] = 0
  camera.from[0] = c.cos(angle) * 6
  --camera.from[1] = -1.0 + 4 * c.sin(angle);
  camera.from[1] = 1.5
  camera.from[2] = c.sin(angle) * 6
  camera.at[0] = 1
  camera.at[1] = 1
  camera.at[2] = 1
  return camera
end



__forbid(__inner)
task renderLoop(
  r : region(ispace(int3d), float),
  colors : ispace(int3d),
  p : partition(disjoint, r, colors)
)
where reads(r)
do
  render.cxx_initialize(__runtime(), __context(), __raw(r), __raw(p),
    __fields(r), 1)

  var stepsPerAngle = 1 -- 100
  var angles = 1 -- 180
  for loop = 0, angles * stepsPerAngle do
    var angle : float = loop * (1.0 / stepsPerAngle)
    var camera = configureCamera(angle)
    render.cxx_render(__runtime(), __context(), camera)
    var direction : float[3]
    for i = 0, 3 do
      direction[i] = camera.at[i] - camera.from[i]
    end
    --render.cxx_saveIndividualImages(__runtime(), __context(), ".")
    render.cxx_reduce(__context(), direction)
    render.cxx_saveImage(__runtime(), __context(), ".")
  end
end



task main()

  -- logical region and partition
  var r = region(ispace(int3d, {2, 2, 2}, {0, 0, 0}), float)
  var colors = ispace(int3d, {2, 2, 2}, {0, 0, 0})
  var p = partition(equal, r, colors)
  fill(r, 0.0)
  renderLoop(r, colors, p)
  render.cxx_terminate()
end

regentlib.saveobj(main, "visualization_2.so", "object", render.cxx_preinitialize)
```

##### cxx_preinitialize
This is called before the runtime starts and performs certain initializations.
visualization_2.rg contains this code:
```
render = terralib.includec("render.h",
{"-I", root_dir,
"-I", runtime_dir,
"-I", mapper_dir,
"-I", legion_dir,
"-I", realm_dir,
})

...

regentlib.saveobj(main, "visualization_2.so", "object", render.cxx_preinitialize)

```

##### cxx_initialize
This entry point is called once when the program starts up.
It creates an ImageCompositor that respects a logical partition that represents the simulation domain.
```
void cxx_initialize(
                   legion_runtime_t runtime_,
                   legion_context_t ctx_,
                   legion_logical_region_t region_,
                   legion_logical_partition_t partition_,
                   legion_field_id_t pFields[],
                   int numPFields
                   )
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion region = CObjectWrapper::unwrap(region_);
  LogicalPartition partition = CObjectWrapper::unwrap(partition_);
  Visualization::ImageDescriptor imageDescriptor = { gImageWidth, gImageHeight, 1 };

  gImageCompositor = new Visualization::ImageReduction(region, partition,
    pFields, numPFields, imageDescriptor, ctx, runtime);
}
```

##### cxx_render
This entry point is called each time the application wants to perform a parallel rendering operation.
The application passes in a camera configuration.
The entry point performs an index launch of the user-supplied render task.
```
void cxx_render(legion_runtime_t runtime_,
                legion_context_t ctx_,
                Camera camera
                ) {
  // Unwrap objects

  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  Visualization::ImageReduction* compositor = gImageCompositor;

  // Setup the render task launch with region requirements
  ArgumentMap argMap;
  ImageDescriptor imageDescriptor = compositor->imageDescriptor();
  size_t argSize = sizeof(ImageDescriptor) + sizeof(camera);
  char args[argSize];
  // ImageDescriptor must be the first argument to the render task
  memcpy(args, (char*)&imageDescriptor, sizeof(ImageDescriptor));
  memcpy(args + sizeof(ImageDescriptor), (char*)&camera, sizeof(camera));
  IndexTaskLauncher renderLauncher(gRenderTaskID, compositor->renderImageDomain(), TaskArgument(args, argSize),
                                   argMap, Predicate::TRUE_PRED, false);

  RegionRequirement req0(imageDescriptor.simulationLogicalPartition, 0, READ_ONLY, EXCLUSIVE,
    imageDescriptor.simulationLogicalRegion);
  for(int i = 0; i < imageDescriptor.numPFields; ++i) {
    req0.add_field(imageDescriptor.pFields[i]);
  }
  renderLauncher.add_region_requirement(req0);

  RegionRequirement req1(compositor->renderImagePartition(), 0, WRITE_DISCARD, EXCLUSIVE,
    compositor->sourceImage());
  req1.add_field(Visualization::ImageReduction::FID_FIELD_R);
  req1.add_field(Visualization::ImageReduction::FID_FIELD_G);
  req1.add_field(Visualization::ImageReduction::FID_FIELD_B);
  req1.add_field(Visualization::ImageReduction::FID_FIELD_A);
  req1.add_field(Visualization::ImageReduction::FID_FIELD_Z);
  req1.add_field(Visualization::ImageReduction::FID_FIELD_USERDATA);
  renderLauncher.add_region_requirement(req1);

  FutureMap futures = runtime->execute_index_space(ctx, renderLauncher);
  futures.wait_all_results();
}
```

##### cxx_reduce
This entry point is called immediately after cxx_render.
It invokes the compositor to perform the image reduction.
```
void cxx_reduce(legion_context_t ctx_, float cameraAt[image_region_dimensions]) {
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  Visualization::ImageReduction* compositor = gImageCompositor;
  compositor->set_blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  compositor->set_blend_equation(GL_FUNC_ADD);
  FutureMap futures = compositor->reduceImages(ctx, cameraAt);
  futures.wait_all_results();
}
```

##### cxx_saveImage
This entry point saves the final composited image to disk.

```
void cxx_saveImage(legion_runtime_t runtime_,
                   legion_context_t ctx_,
                   const char* outDir
                   ) {
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  // save the image
  Visualization::ImageReduction* compositor = gImageCompositor;
  ImageDescriptor imageDescriptor = compositor->imageDescriptor();
  size_t argLen = sizeof(ImageDescriptor) + strlen(outDir) + 1;
  char args[argLen] = { 0 };
  memcpy(args, &imageDescriptor, sizeof(imageDescriptor));
  strcpy(args + sizeof(imageDescriptor), outDir);
  TaskLauncher saveImageLauncher(gSaveImageTaskID, TaskArgument(args, argLen), Predicate::TRUE_PRED);
  DomainPoint slice0 = Legion::Point<3>::ZEROES();
  LogicalRegion imageSlice0 = runtime->get_logical_subregion_by_color(compositor->compositeImagePartition(), slice0);
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


## Nonassociative todo
Associative reductions can be implemented in a tree.
Non-associative reductions must be implemented in a serial chain.
Non-commutative reductions subsume commutative reductions
so it is sufficient to implement only the non-commutative form.

These can be implemented when there is a need.
