//
// render.cc
//
// example C++ interface to support visualization
//

#include "image.h"
#include "render.h"
#include "legion_visualization.h"

#include <sstream>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkCPProcessor.h>
#include <vtkCPPythonScriptPipeline.h>
#include <vtkCellData.h>
#include <vtkCellType.h>
#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkImageData.h>
#include <vtkXMLImageDataReader.h>

#define _T {std::cout<<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<std::endl;}

using ImageReduction = Legion::Visualization::ImageReduction;
using ImageDescriptor = Legion::Visualization::ImageDescriptor;
using namespace Legion;

// global data
Legion::MapperID imageReductionMapperID = 1;
static ImageReduction* gImageCompositor = nullptr;
static vtkCPProcessor* VTKProcessor = NULL;
static vtkImageData* VTKGrid = NULL;
static int gRenderTaskID = 0;
static int gSaveImageTaskID = 0;
static int gFrameNumber = 0;
static int gImageWidth = 2430;
static int gImageHeight = 1180;

void legion_wait_on_mpi()
{
  handshake.legion_wait_on_mpi();
}

void legion_handoff_to_mpi()
{
  handshake.legion_handoff_to_mpi();
}

static void render_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  PhysicalRegion data = regions[0];
  PhysicalRegion image = regions[1];
  IndexSpace indexSpace = data.get_logical_region().get_index_space();
  Domain domain = runtime->get_index_space_domain(ctx, indexSpace);
  Legion::Rect<3> bounds = runtime->get_index_space_domain(ctx, indexSpace);

  LogicalRegion dataRegion = data.get_logical_region();
  FieldSpace fspace = dataRegion.get_field_space();
  RegionRequirement dataReq = task->regions[0];
  int rank = runtime->find_local_MPI_rank();
  char* argsPtr = (char*)task->args;
  ImageDescriptor* imageDescriptor = (ImageDescriptor*)(argsPtr);
  Camera* camera = (Camera*)(argsPtr + sizeof(ImageDescriptor));
  int* timestep = (int*)(argsPtr + sizeof(ImageDescriptor) + sizeof(Camera));
  printf("RANK: %d TIMESTEP: %d ", rank, *timestep);
  printf("Lo: %d %d %d Hi: %d %d %d\n",
         bounds.lo[0], bounds.lo[1], bounds.lo[2],
         bounds.hi[0], bounds.hi[1], bounds.hi[2]);

  if(VTKGrid == NULL)
  {
    VTKGrid = vtkImageData::New();
    VTKGrid->SetExtent(bounds.lo[0], bounds.hi[0],
                       bounds.lo[1], bounds.hi[1],
                       bounds.lo[2], bounds.hi[2]);
    VTKGrid->SetSpacing(1000, 1000, 1000);
    VTKGrid->SetOrigin(bounds.lo[0], bounds.lo[1], bounds.lo[2]);
  }

  vtkNew<vtkCPDataDescription> dataDescription;
  dataDescription->SetTimeData((*timestep)*0.1, *timestep);

  for(std::set<FieldID>::iterator it = dataReq.privilege_fields.begin();
      it != dataReq.privilege_fields.end(); ++it)
  {
    FieldID fid = *it;
    const char *field_name;
    runtime->retrieve_name(fspace, fid, field_name);
    dataDescription->AddInput(field_name);
  }

  if (VTKProcessor->RequestDataDescription(dataDescription.GetPointer()) != 0)
  {
    for(std::set<FieldID>::iterator it = dataReq.privilege_fields.begin();
        it != dataReq.privilege_fields.end(); ++it)
    {
      FieldID fid = *it;
      const char *field_name;
      runtime->retrieve_name(fspace, fid, field_name);
      vtkCPInputDataDescription* idd = dataDescription->GetInputDescriptionByName(field_name);
      if (idd->IsFieldNeeded(field_name, vtkDataObject::POINT) == true)
      {
        if (VTKGrid->GetPointData()->GetNumberOfArrays() == 0)
        {
          vtkNew<vtkDoubleArray> arr;
          arr->SetName(field_name);
          arr->SetNumberOfComponents(1);
          arr->SetNumberOfTuples(static_cast<vtkIdType>(domain.get_volume()));
          VTKGrid->GetPointData()->AddArray(arr.GetPointer());
        }

        vtkDoubleArray* arr = vtkDoubleArray::SafeDownCast(VTKGrid->GetPointData()->GetArray(field_name));
        AccessorRO<double, 3> data_acc(data, *it);
        arr->SetArray(data_acc.ptr(bounds.lo), domain.get_volume(), 1);
      }
      idd->SetGrid(VTKGrid);
    }
    VTKProcessor->CoProcess(dataDescription.GetPointer());
  }

  PNGImage *pngimage = new PNGImage();
  std::stringstream ss;
  ss << "rank" << rank << "/RenderView1_" << *timestep << ".png";
  read_png_file(ss.str().c_str(), pngimage);

  ss.str(std::string());
  ss.clear();
  ss << "rank" << rank << "/z_buffer_" << *timestep << ".vti";
  vtkXMLImageDataReader *reader = vtkXMLImageDataReader::New();
  reader->SetFileName(ss.str().c_str());
  reader->Update();
  vtkImageData *buffer = reader->GetOutput();
  vtkFloatArray *z_buf = vtkFloatArray::SafeDownCast(buffer->GetPointData()->GetArray(0));

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

  for(PointInRectIterator<3> pir(saveRect); pir(); pir++) {
    DomainPoint point(*pir);
    int x = point[0];
    int y = point[1];
    r[*pir] = pngimage->R(x, y);
    g[*pir] = pngimage->G(x, y);
    b[*pir] = pngimage->B(x, y);
    a[*pir] = pngimage->A(x, y);
    z[*pir] = *(z_buf->GetTuple(x * gImageHeight + y));
    u[*pir] = 0; // user defined channel, unused
  }

  delete pngimage;
}

static int save_image_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
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
void cxx_preinitialize()
{
  ImageReduction::preinitializeBeforeRuntimeStarts();
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
  gRenderTaskID = Legion::Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(gRenderTaskID, "render_task");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
    .add_layout_constraint_set(0/*index*/, soa_layout_id)
    .add_layout_constraint_set(1/*index*/, soa_layout_id)
    .add_layout_constraint_set(2/*index*/, soa_layout_id);
  Runtime::preregister_task_variant<render_task>(registrar, "render_task");

  // preregister save image task
  gSaveImageTaskID = Legion::Runtime::generate_static_task_id();
  TaskVariantRegistrar registrarSaveImage(gSaveImageTaskID, "save_image_task");
  registrarSaveImage.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
    .add_layout_constraint_set(0/*index*/, soa_layout_id);
  Runtime::preregister_task_variant<int, save_image_task>(registrarSaveImage,
                                                          "save_image_task");
}

// this entry point is called once from the main task
void cxx_initialize(legion_runtime_t runtime_,
                    legion_context_t ctx_,
                    legion_logical_region_t region_,
                    legion_logical_partition_t partition_,
                    legion_field_id_t pFields[],
                    int numPFields)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  LogicalRegion region = CObjectWrapper::unwrap(region_);
  LogicalPartition partition = CObjectWrapper::unwrap(partition_);
  ImageDescriptor imageDescriptor = { gImageWidth, gImageHeight, 1 };
  gImageCompositor = new ImageReduction(region, partition,
                                        pFields, numPFields, imageDescriptor, ctx, runtime,
                                        imageReductionMapperID);
  int rank = runtime->find_local_MPI_rank();

  if (VTKProcessor == NULL)
  {
    std::stringstream ss;
    ss << "rank" << rank;
    VTKProcessor = vtkCPProcessor::New();
    VTKProcessor->Initialize(ss.str().c_str());
  }
  else
  {
    VTKProcessor->RemoveAllPipelines();
  }

  const InputArgs &args = Runtime::get_input_args();

  for (int i = 0; i < args.argc; i++)
  {
    if (!strcmp(args.argv[i], "-pipeline"))
    {
      vtkNew<vtkCPPythonScriptPipeline> pipeline;
      pipeline->Initialize(args.argv[i+1]);
      VTKProcessor->AddPipeline(pipeline.GetPointer());
    }
  }
}

// this entry point is called once from the main task
void cxx_render(legion_runtime_t runtime_,
                legion_context_t ctx_,
                Camera camera,
                int timestep)
{
  // Unwrap objects

  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  ImageReduction* compositor = gImageCompositor;

  // Setup the render task launch with region requirements
  ArgumentMap argMap;
  ImageDescriptor imageDescriptor = compositor->imageDescriptor();
  size_t argSize = sizeof(ImageDescriptor) + sizeof(camera) + sizeof(int);
  char args[argSize];
  // ImageDescriptor must be the first argument to the render task
  memcpy(args, (char*)&imageDescriptor, sizeof(ImageDescriptor));
  memcpy(args + sizeof(ImageDescriptor), (char*)&camera, sizeof(camera));
  memcpy(args + sizeof(ImageDescriptor) + sizeof(camera), (int*)&timestep, sizeof(timestep));

  IndexTaskLauncher renderLauncher(gRenderTaskID, compositor->renderImageDomain(), TaskArgument(args, argSize),
                                   argMap, Predicate::TRUE_PRED, false, imageReductionMapperID);

  RegionRequirement req0(imageDescriptor.simulationLogicalPartition, 0, READ_ONLY, EXCLUSIVE,
                         imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  for(int i = 0; i < imageDescriptor.numPFields; ++i) {
    req0.add_field(imageDescriptor.pFields[i]);
  }
  renderLauncher.add_region_requirement(req0);

  RegionRequirement req1(compositor->renderImagePartition(), 0, WRITE_DISCARD, EXCLUSIVE,
                         compositor->sourceImage(), imageReductionMapperID);
  req1.add_field(ImageReduction::FID_FIELD_R);
  req1.add_field(ImageReduction::FID_FIELD_G);
  req1.add_field(ImageReduction::FID_FIELD_B);
  req1.add_field(ImageReduction::FID_FIELD_A);
  req1.add_field(ImageReduction::FID_FIELD_Z);
  req1.add_field(ImageReduction::FID_FIELD_USERDATA);
  renderLauncher.add_region_requirement(req1);

  runtime->execute_index_space(ctx, renderLauncher);
}

void cxx_reduce(legion_context_t ctx_,
                Camera camera)
{
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  ImageReduction* compositor = gImageCompositor;
  compositor->set_blend_func(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  compositor->set_blend_equation(GL_FUNC_ADD);

  float cameraDirection[] = {
    (float)(camera.at[0] - camera.from[0]),
    (float)(camera.at[1] - camera.from[1]),
    (float)(camera.at[2] - camera.from[2])
  };

  compositor->reduceImages(ctx, cameraDirection);
}

void cxx_saveImage(legion_runtime_t runtime_,
                   legion_context_t ctx_,
                   const char* outDir)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  // save the image
  ImageReduction* compositor = gImageCompositor;
  ImageDescriptor imageDescriptor = compositor->imageDescriptor();
  size_t argLen = sizeof(ImageDescriptor) + strlen(outDir) + 1;
  char args[argLen] = { 0 };
  memcpy(args, &imageDescriptor, sizeof(imageDescriptor));
  strcpy(args + sizeof(imageDescriptor), outDir);
  TaskLauncher saveImageLauncher(gSaveImageTaskID, TaskArgument(args, argLen), Predicate::TRUE_PRED,
                                 imageReductionMapperID);
  DomainPoint slice0 = Legion::Point<3>::ZEROES();
  LogicalRegion imageSlice0 = runtime->get_logical_subregion_by_color(compositor->compositeImagePartition(),
                                                                      slice0);
  RegionRequirement req(imageSlice0, READ_ONLY, EXCLUSIVE, compositor->sourceImage(),
                        imageReductionMapperID);
  saveImageLauncher.add_region_requirement(req);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_R);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_G);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_B);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_A);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_Z);
  runtime->execute_task(ctx, saveImageLauncher);
}

// this is provided for debugging the renderer
void cxx_saveIndividualImages(legion_runtime_t runtime_,
                              legion_context_t ctx_,
                              const char* outDir)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  // save the image
  ImageReduction* compositor = gImageCompositor;
  ImageDescriptor imageDescriptor = compositor->imageDescriptor();
  ArgumentMap argMap;
  size_t argLen = sizeof(ImageDescriptor) + strlen(outDir) + 1;
  char args[argLen] = { 0 };
  memcpy(args, &imageDescriptor, sizeof(imageDescriptor));
  strcpy(args + sizeof(imageDescriptor), outDir);
  IndexTaskLauncher saveImageLauncher(gSaveImageTaskID,
                                      compositor->compositeImageDomain(),
                                      TaskArgument(args, argLen),
                                      argMap, Predicate::TRUE_PRED, false,
                                      imageReductionMapperID);
  RegionRequirement req(compositor->compositeImagePartition(), 0, READ_ONLY, EXCLUSIVE,
                        compositor->sourceImage(),
                        imageReductionMapperID);
  saveImageLauncher.add_region_requirement(req);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_R);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_G);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_B);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_A);
  saveImageLauncher.add_field(0/*idx*/, ImageReduction::FID_FIELD_Z);
  runtime->execute_index_space(ctx, saveImageLauncher);
}

void cxx_terminate()
{
  delete gImageCompositor;
}

