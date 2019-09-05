//
// render.cc
//
// example C++ interface to support visualization
//


#include "render.h"
#include "legion_visualization.h"
#include "image_reduction_mapper.h"


#ifdef __cplusplus
extern "C" {
#endif
  
  
  
  // global data
  
  static Visualization::ImageReduction* gImageCompositor = nullptr;
  static legion_mapper_id_t gImageReductionMapperID = 0;
  static int gRenderTaskID = 0;
  static int gSaveImageTaskID = 0;
  static int gFrameNumber = 0; 
  static int gImageWidth = 1280;
  static int gImageHeight = 720;
  
  
  void renderCube(Rect<3> bounds, ImageDescriptor* imageDescriptor, Camera* camera, unsigned char*& rgbaBuffer, float*& depthBuffer);
  
  static void render_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, HighLevelRuntime *runtime) {
    
    PhysicalRegion data = regions[0];
    PhysicalRegion image = regions[1];
    IndexSpace indexSpace = data.get_logical_region().get_index_space();
    Rect<3> bounds = runtime->get_index_space_domain(ctx, indexSpace);
    char* argsPtr = (char*)task->args;
    ImageDescriptor* imageDescriptor = (ImageDescriptor*)(argsPtr);
    Camera* camera = (Camera*)(argsPtr + sizeof(ImageDescriptor));
    unsigned char* rgbaBuffer;
    float* depthBuffer;
    
    // draw a cube in the center of the space
    
    renderCube(bounds, imageDescriptor, camera, rgbaBuffer, depthBuffer);
    
    // now copy the image data into the image logical region
    
    glReadPixels(0, 0, imageDescriptor->width, imageDescriptor->height, GL_DEPTH_COMPONENT, GL_FLOAT, depthBuffer);
    
    std::vector<legion_field_id_t> imageFields;
    image.get_fields(imageFields);
    AccessorWO<ImageReduction::PixelField, 3> r(image, imageFields[0]);
    AccessorWO<ImageReduction::PixelField, 3> g(image, imageFields[1]);
    AccessorWO<ImageReduction::PixelField, 3> b(image, imageFields[2]);
    AccessorWO<ImageReduction::PixelField, 3> a(image, imageFields[3]);
    AccessorWO<ImageReduction::PixelField, 3> z(image, imageFields[4]);
    AccessorWO<ImageReduction::PixelField, 3> u(image, imageFields[5]);

    IndexSpace saveIndexSpace = image.get_logical_region().get_index_space();
    Rect<3> saveRect = runtime->get_index_space_domain(ctx, saveIndexSpace);
    
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

    delete [] rgbaBuffer;
    delete [] depthBuffer;
  }
  
  
  
  static int save_image_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, HighLevelRuntime *runtime) {
    ImageDescriptor* imageDescriptor = (ImageDescriptor*)(task->args);
    unsigned char* outDir = (unsigned char*)(task->args) + sizeof(ImageDescriptor);
    PhysicalRegion image = regions[0];
    std::vector<legion_field_id_t> imageFields;
    image.get_fields(imageFields);
    
    AccessorRO<ImageReduction::PixelField, 3> r(image, imageFields[0]);
    AccessorRO<ImageReduction::PixelField, 3> g(image, imageFields[1]);
    AccessorRO<ImageReduction::PixelField, 3> b(image, imageFields[2]);
    AccessorRO<ImageReduction::PixelField, 3> a(image, imageFields[3]);
    AccessorRO<ImageReduction::PixelField, 3> z(image, imageFields[4]);
    
    char filename[1024];
    sprintf(filename, "%s/image.%05d.tga", outDir, gFrameNumber++);
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
    Rect<3> saveRect = runtime->get_index_space_domain(ctx, saveIndexSpace);
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
  void cxx_preinitialize(legion_mapper_id_t mapperID) {
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
    gImageReductionMapperID = mapperID;
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
    Runtime::preregister_task_variant<int, save_image_task>(registrarSaveImage, "save_image_task");
    
  }
  
  
  
  
  
  
  
  
  // this entry point is called once from the main task
  RegionPartition cxx_initialize(
                                 legion_runtime_t runtime_,
                                 legion_context_t ctx_,
                                 legion_logical_partition_t partition_
                                 )
  {
    Runtime *runtime = CObjectWrapper::unwrap(runtime_);
    Context ctx = CObjectWrapper::unwrap(ctx_)->context();
    LogicalPartition partition = CObjectWrapper::unwrap(partition_);
    
    // Initialize an image compositor, or reuse an initialized one
    
    Visualization::ImageDescriptor imageDescriptor = { gImageWidth, gImageHeight, 1 };
    
    if(gImageCompositor == nullptr) {
      gImageCompositor = new Visualization::ImageReduction(partition, imageDescriptor, ctx, runtime, gImageReductionMapperID);
      ImageReductionMapper::registerRenderTaskName("render_task");
    }
    
    Visualization::ImageReduction* compositor = gImageCompositor;
    RegionPartition result;
    result.indexSpace = CObjectWrapper::wrap(compositor->sourceIndexSpace());
    result.imageX = CObjectWrapper::wrap(compositor->sourceImage());
    result.colorSpace = CObjectWrapper::wrap(compositor->everywhereColorSpace());
    result.p_Image = CObjectWrapper::wrap(compositor->everywherePartition());
    compositor->sourceImageFields(ctx, result.imageFields);
    return result;
  }
  
  
  
  
  
  
  
  // this entry point is called once from the main task
  void cxx_render(legion_runtime_t runtime_,
                  legion_context_t ctx_,
                  legion_physical_region_t image_[],
                  legion_field_id_t imageFields[],
                  int numImageFields,
                  legion_logical_region_t r_,
                  legion_logical_partition_t p_,
                  legion_field_id_t pFields[],
                  int numPFields,
                  Camera camera
                  ) {

    static bool firstTime = true;
    
    // Unwrap objects
    
    Runtime *runtime = CObjectWrapper::unwrap(runtime_);
    Context ctx = CObjectWrapper::unwrap(ctx_)->context();
    PhysicalRegion* image = CObjectWrapper::unwrap(image_[0]);
    LogicalRegion r = CObjectWrapper::unwrap(r_);
    LogicalPartition p = CObjectWrapper::unwrap(p_);

    
    // Create projection functors
    
    Visualization::ImageReduction* compositor = gImageCompositor;
    if(firstTime) {
      ImageReductionProjectionFunctor* functor0 = new ImageReductionProjectionFunctor(compositor->everywhereDomain(), p);
      runtime->register_projection_functor(1, functor0);
      ImageReductionProjectionFunctor* functor1 = new ImageReductionProjectionFunctor(compositor->everywhereDomain(), compositor->everywherePartition());
      runtime->register_projection_functor(2, functor1);
    }
    
    // Setup the render task launch with region requirements
    
    ArgumentMap argMap;
    ImageDescriptor imageDescriptor = compositor->imageDescriptor();
    size_t argSize = sizeof(ImageDescriptor) + sizeof(camera);
    char args[argSize];
    memcpy(args, (char*)&imageDescriptor, sizeof(ImageDescriptor));
    memcpy(args + sizeof(ImageDescriptor), (char*)&camera, sizeof(camera));
    IndexTaskLauncher renderLauncher(gRenderTaskID, compositor->everywhereDomain(), TaskArgument(args, argSize),
                                     argMap, Predicate::TRUE_PRED, false, gImageReductionMapperID);
    
    RegionRequirement req0(p, 1, READ_ONLY, EXCLUSIVE, r, gImageReductionMapperID);
    for(int i = 0; i < numPFields; ++i) {
      req0.add_field(pFields[i]);
    }
    renderLauncher.add_region_requirement(req0);

    RegionRequirement req1(compositor->everywherePartition(), 3, WRITE_DISCARD, EXCLUSIVE, image->get_logical_region(), gImageReductionMapperID);
    req1.add_field(Visualization::ImageReduction::FID_FIELD_R);
    req1.add_field(Visualization::ImageReduction::FID_FIELD_G);
    req1.add_field(Visualization::ImageReduction::FID_FIELD_B);
    req1.add_field(Visualization::ImageReduction::FID_FIELD_A);
    req1.add_field(Visualization::ImageReduction::FID_FIELD_Z);
    req1.add_field(Visualization::ImageReduction::FID_FIELD_USERDATA);
    renderLauncher.add_region_requirement(req1);
    
    // Clear the mapping history so render_task will create it anew
    
    ImageReductionMapper::clearPlacement(p);
    
    // Launch the render task
    
    FutureMap futures = runtime->execute_index_space(ctx, renderLauncher);
    futures.wait_all_results();
    firstTime = false;
  }
  
  
  void cxx_reduce(legion_context_t ctx_) {

    Context ctx = CObjectWrapper::unwrap(ctx_)->context();
    Visualization::ImageReduction* compositor = gImageCompositor;
    compositor->set_depth_func(GL_LESS);
    FutureMap futures = compositor->reduce_associative_noncommutative(ctx);
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
  
  
  
#ifdef __cplusplus
}
#endif
