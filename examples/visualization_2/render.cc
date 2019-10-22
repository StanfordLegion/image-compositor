//
// render.cc
//
// example C++ interface to support visualization
//


#include "render.h"
#include "legion_visualization.h"
#include "GL/osmesa.h"


#define _T {std::cout<<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<std::endl;}

#ifdef __cplusplus
extern "C" {
#endif

  using namespace Legion;

  // global data

  static Visualization::ImageReduction* gImageCompositor = nullptr;
  static int gRenderTaskID = 0;
  static int gSaveImageTaskID = 0;
  static int gFrameNumber = 0;
  static int gImageWidth = 1280;
  static int gImageHeight = 720;


  void createGraphicsContext(OSMesaContext &mesaCtx,
                             GLubyte* &rgbaBuffer,
                             GLfloat* &depthBuffer,
                             int width,
                             int height);

  void renderCube(Legion::Rect<3> bounds, ImageDescriptor* imageDescriptor,
    Camera* camera, unsigned char*& rgbaBuffer, float*& depthBuffer);

  static void render_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime) {
    PhysicalRegion data = regions[0];
    PhysicalRegion image = regions[1];
    IndexSpace indexSpace = data.get_logical_region().get_index_space();
    Legion::Rect<3> bounds = runtime->get_index_space_domain(ctx, indexSpace);
    char* argsPtr = (char*)task->args;

    ImageDescriptor* imageDescriptor = (ImageDescriptor*)(argsPtr);
    Camera* camera = (Camera*)(argsPtr + sizeof(ImageDescriptor));

    // draw a cube in the center of the space

    OSMesaContext mesaCtx;
    unsigned char* rgbaBuffer;
    float* depthBuffer;
    createGraphicsContext(mesaCtx, rgbaBuffer, depthBuffer,
      imageDescriptor->width, imageDescriptor->height);

    renderCube(bounds, imageDescriptor, camera, rgbaBuffer, depthBuffer);

#if 0
{
  for(unsigned i = 0; i < 1280 * 720; ++i) {
    unsigned char r = rgbaBuffer[i * 4 + 0];
    unsigned char g = rgbaBuffer[i * 4 + 1];
    unsigned char b = rgbaBuffer[i * 4 + 2];
    unsigned char alpha = rgbaBuffer[i * 4 + 3];
    char buffer[256];
    if(r != 0 || g != 0 || b != 0) {
      sprintf(buffer, "%d RGBA %d %d %d %d\n", i, r, g, b, alpha);
      std::cout << buffer;
    }
  }
}
#endif

    // now copy the image data into the image logical region

    glReadPixels(0, 0, imageDescriptor->width, imageDescriptor->height,
      GL_DEPTH_COMPONENT, GL_FLOAT, depthBuffer);

    std::vector<legion_field_id_t> imageFields;
    image.get_fields(imageFields);
    AccessorWO<ImageReduction::PixelField, 3> r(image, imageFields[0]);
    AccessorWO<ImageReduction::PixelField, 3> g(image, imageFields[1]);
    AccessorWO<ImageReduction::PixelField, 3> b(image, imageFields[2]);
    AccessorWO<ImageReduction::PixelField, 3> a(image, imageFields[3]);
    AccessorWO<ImageReduction::PixelField, 3> z(image, imageFields[4]);
    AccessorWO<ImageReduction::PixelField, 3> u(image, imageFields[5]);

    IndexSpace saveIndexSpace = image.get_logical_region().get_index_space();
    Legion::Rect<3> saveRect = runtime->get_index_space_domain(ctx, saveIndexSpace);

    int index = 0;
    for(PointInRectIterator<3> pir(saveRect); pir(); pir++) {
      r[*pir] = rgbaBuffer[index * 4] / 255.0;
      g[*pir] = rgbaBuffer[index * 4 + 1] / 255.0;
      b[*pir] = rgbaBuffer[index * 4 + 2] / 255.0;
      a[*pir] = rgbaBuffer[index * 4 + 3] / 255.0;
      z[*pir] = depthBuffer[index];
      u[*pir] = 0; // user defined channel, unused
      index++;
    }

    OSMesaDestroyContext(mesaCtx);
    delete [] rgbaBuffer;
    delete [] depthBuffer;
  }



  static int save_image_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, HighLevelRuntime *runtime) {
    ImageDescriptor* imageDescriptor = (ImageDescriptor*)(task->args);
    unsigned char* outDir = (unsigned char*)(task->args) + sizeof(ImageDescriptor);
    PhysicalRegion image = regions[0];
    IndexSpace indexSpace = image.get_logical_region().get_index_space();
    Legion::Rect<3> bounds = runtime->get_index_space_domain(ctx, indexSpace);
    std::vector<legion_field_id_t> imageFields;
    image.get_fields(imageFields);

    AccessorRO<ImageReduction::PixelField, 3> r(image, imageFields[0]);
    AccessorRO<ImageReduction::PixelField, 3> g(image, imageFields[1]);
    AccessorRO<ImageReduction::PixelField, 3> b(image, imageFields[2]);
    AccessorRO<ImageReduction::PixelField, 3> a(image, imageFields[3]);
    AccessorRO<ImageReduction::PixelField, 3> z(image, imageFields[4]);

    char filename[1024];
    sprintf(filename, "%s/image_%d_%d_%d.%05d.tga", outDir, (
      int)bounds.lo.x, (int)bounds.lo.y, (int)bounds.lo.z, gFrameNumber++);
    FILE* f = fopen(filename, "w");
    if(f == nullptr) {
      std::cerr << "could not create file " << filename << std::endl;
      return -1;
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
    fputc (imageDescriptor->width & 0xff, f);      /* Image Width */
    fputc ((imageDescriptor->width>>8) & 0xff, f);
    fputc (imageDescriptor->height & 0xff, f);     /* Image Height   */
    fputc ((imageDescriptor->height>>8) & 0xff, f);
    fputc (0x18, f);     /* Pixel Depth, 0x18 => 24 Bits  */
    fputc (0x20, f);     /* Image Descriptor  */
    fclose(f);

    f = fopen(filename, "ab");  /* reopen in binary append mode */
    IndexSpace saveIndexSpace = image.get_logical_region().get_index_space();
    Legion::Rect<3> saveRect = runtime->get_index_space_domain(ctx, saveIndexSpace);
    PointInRectIterator<3> pir(saveRect);
    ImageReduction::PixelField* BB = (ImageReduction::PixelField*)b.ptr(*pir);
    ImageReduction::PixelField* GG = (ImageReduction::PixelField*)g.ptr(*pir);
    ImageReduction::PixelField* RR = (ImageReduction::PixelField*)r.ptr(*pir);

    for(int y = imageDescriptor->height - 1; y >= 0; y--) {
      for(int x = 0; x < imageDescriptor->width; ++x) {
        int index = x + y * imageDescriptor->width;
        GLubyte b_ = BB[index] * 255;
        fputc(b_, f); /* write blue */
        GLubyte g_ = GG[index] * 255;
        fputc(g_, f); /* write green */
        GLubyte r_ = RR[index] * 255;
        fputc(r_, f);   /* write red */
      }
    }
    fclose(f);
    std::cout << "wrote image " << filename << std::endl;
    return 0;
  }








  // Called from mapper before runtime has started
  void cxx_preinitialize() {
    Visualization::ImageReduction::preinitializeBeforeRuntimeStarts();
    // allocate physical regions contiguously in memory
    LayoutConstraintRegistrar layout_registrar(FieldSpace::NO_SPACE, "SOA layout");
    std::vector<DimensionKind> dim_order(4);
    dim_order[0] = DIM_X;
    dim_order[1] = DIM_Y;
    dim_order[2] = DIM_Z;
    dim_order[3] = DIM_F; // fields go last for SOA
    layout_registrar.add_constraint(OrderingConstraint(dim_order, true/*contig*/));
    LayoutConstraintID soa_layout_id = Runtime::preregister_layout(layout_registrar);


    // preregister render task
    gRenderTaskID = Legion::HighLevelRuntime::generate_static_task_id();
    TaskVariantRegistrar registrar(gRenderTaskID, "render_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
    .add_layout_constraint_set(0/*index*/, soa_layout_id)
    .add_layout_constraint_set(1/*index*/, soa_layout_id)
    .add_layout_constraint_set(2/*index*/, soa_layout_id);
    Runtime::preregister_task_variant<render_task>(registrar, "render_task");

    // preregister save image task
    gSaveImageTaskID = Legion::HighLevelRuntime::generate_static_task_id();
    TaskVariantRegistrar registrarSaveImage(gSaveImageTaskID, "save_image_task");
    registrarSaveImage.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
    .add_layout_constraint_set(0/*index*/, soa_layout_id);
    Runtime::preregister_task_variant<int, save_image_task>(registrarSaveImage,
      "save_image_task");
  }








  // this entry point is called once from the main task
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







  // this entry point is called once from the main task
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






  void cxx_reduce(legion_context_t ctx_, float cameraDirection[image_region_dimensions]) {
    Context ctx = CObjectWrapper::unwrap(ctx_)->context();
    Visualization::ImageReduction* compositor = gImageCompositor;
    compositor->set_blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    compositor->set_blend_equation(GL_FUNC_ADD);
    FutureMap futures = compositor->reduceImages(ctx, cameraDirection);
    futures.wait_all_results();
  }


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


  // this is provided for debugging the renderer

  void cxx_saveIndividualImages(legion_runtime_t runtime_,
                                legion_context_t ctx_,
                                const char* outDir
                                ) {
    Runtime *runtime = CObjectWrapper::unwrap(runtime_);
    Context ctx = CObjectWrapper::unwrap(ctx_)->context();

    // save the image
    Visualization::ImageReduction* compositor = gImageCompositor;
    ImageDescriptor imageDescriptor = compositor->imageDescriptor();
    ArgumentMap argMap;
    size_t argLen = sizeof(ImageDescriptor) + strlen(outDir) + 1;
    char args[argLen] = { 0 };
    memcpy(args, &imageDescriptor, sizeof(imageDescriptor));
    strcpy(args + sizeof(imageDescriptor), outDir);
    IndexTaskLauncher saveImageLauncher(gSaveImageTaskID, compositor->compositeImageDomain(), TaskArgument(args, argLen),
                                     argMap, Predicate::TRUE_PRED, false);
    RegionRequirement req(compositor->compositeImagePartition(), 0, READ_ONLY, EXCLUSIVE, compositor->sourceImage());
    saveImageLauncher.add_region_requirement(req);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_R);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_G);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_B);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_A);
    saveImageLauncher.add_field(0/*idx*/, Visualization::ImageReduction::FID_FIELD_Z);
    FutureMap futures = runtime->execute_index_space(ctx, saveImageLauncher);
    futures.wait_all_results();
  }


  void cxx_terminate() {
    delete gImageCompositor;
  }

#ifdef __cplusplus
}
#endif
