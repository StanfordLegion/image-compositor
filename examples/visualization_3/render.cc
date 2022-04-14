//
// render.cc
//
// example C++ interface to support visualization
//

#include "image.h"
#include "render.h"
#include "legion_visualization.h"

#include <sstream>
#include <cstdio>
// #include <vtkCPDataDescription.h>
// #include <vtkCPInputDataDescription.h>
// #include <vtkCPProcessor.h>
// #include <vtkCPPythonScriptPipeline.h>
// #include <vtkCellData.h>
// #include <vtkCellType.h>
// #include <vtkDoubleArray.h>
// #include <vtkFloatArray.h>
// #include <vtkNew.h>
// #include <vtkPointData.h>
// #include <vtkPoints.h>
// #include <vtkImageData.h>
// #include <vtkXMLImageDataReader.h>

#include "s3d_projection.h"

#include "common/imageio.h"
#include "renderer.h"

#define _T {std::cout<<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<std::endl;}

using ImageReduction = Legion::Visualization::ImageReduction;
using ImageDescriptor = Legion::Visualization::ImageDescriptor;
using namespace Legion;

// global data
Legion::MapperID imageReductionMapperID = 1;
static ImageReduction* gImageCompositor = nullptr;
// static vtkCPProcessor* VTKProcessor = NULL;
// static vtkImageData* VTKGrid = NULL;
static int gRenderTaskID = 0;
static int gSaveImageTaskID = 0;
static int gImageWidth  = 800; // 2430;
static int gImageHeight = 600; // 1180;

struct SaveImageArgs {
  ImageDescriptor imageDescriptor;
  char *outdir;
  int timestep;

  SaveImageArgs(ImageDescriptor imageDescriptor, char *outdir, int timestep) :
    imageDescriptor(imageDescriptor),
    outdir(outdir),
    timestep(timestep)
  {};
};

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
  const int rank = runtime->find_local_MPI_rank();

  PhysicalRegion data  = regions[0];
  PhysicalRegion image = regions[1];

  IndexSpace indexSpace = data.get_logical_region().get_index_space();
  Domain domain = runtime->get_index_space_domain(ctx, indexSpace);
  Legion::Rect<3> bounds = domain;

  LogicalRegion dataRegion = data.get_logical_region();
  FieldSpace fspace = dataRegion.get_field_space();
  RegionRequirement dataReq = task->regions[0];

  // Access ghost regions
  RegionRequirement gReqXm = task->regions[2];
  RegionRequirement gReqXp = task->regions[3];
  RegionRequirement gReqYm = task->regions[4];
  RegionRequirement gReqYp = task->regions[5];
  RegionRequirement gReqZm = task->regions[6];
  RegionRequirement gReqZp = task->regions[7];

  PhysicalRegion gDataXm = regions[2];
  PhysicalRegion gDataXp = regions[3];
  PhysicalRegion gDataYm = regions[4];
  PhysicalRegion gDataYp = regions[5];
  PhysicalRegion gDataZm = regions[6];
  PhysicalRegion gDataZp = regions[7];

  // Access camera information
  char* argsPtr = (char*)task->args;
  ImageDescriptor* imageDescriptor = (ImageDescriptor*)(argsPtr);
  Camera* camera = (Camera*)(argsPtr + sizeof(ImageDescriptor));
  int* timestep = (int*)(argsPtr + sizeof(ImageDescriptor) + sizeof(Camera));

  printf("[render] RANK: %d TIMESTEP: %d\n"
         "         Lo: %lld %lld %lld Hi: %lld %lld %lld\n"
         "         Camera Up: %f %f %f, At: %f %f %f, From: %f %f %f\n", 
         rank, *timestep,
         bounds.lo[0], bounds.lo[1], bounds.lo[2],
         bounds.hi[0], bounds.hi[1], bounds.hi[2],
         camera->up[0],   camera->up[1],   camera->up[2],
         camera->at[0],   camera->at[1],   camera->at[2],
         camera->from[0], camera->from[1], camera->from[2]);

  // if(VTKGrid == NULL)
  // {
  //   VTKGrid = vtkImageData::New();
  //   VTKGrid->SetExtent(bounds.lo[0], bounds.hi[0],
  //                      bounds.lo[1], bounds.hi[1],
  //                      bounds.lo[2], bounds.hi[2]);
  //   VTKGrid->SetSpacing(1000, 1000, 1000);
  //   VTKGrid->SetOrigin(bounds.lo[0], bounds.lo[1], bounds.lo[2]);
  // }
  //
  // vtkNew<vtkCPDataDescription> dataDescription;
  // dataDescription->SetTimeData((*timestep)*0.1, *timestep);
  //
  // for(std::set<FieldID>::iterator it = dataReq.privilege_fields.begin();
  //     it != dataReq.privilege_fields.end(); ++it)
  // {
  //   FieldID fid = *it;
  //   const char *field_name;
  //   runtime->retrieve_name(fspace, fid, field_name);
  //   dataDescription->AddInput(field_name);
  // }
  //
  // if (VTKProcessor->RequestDataDescription(dataDescription.GetPointer()) != 0)
  // {
  //   for(std::set<FieldID>::iterator it = dataReq.privilege_fields.begin();
  //       it != dataReq.privilege_fields.end(); ++it)
  //   {
  //     FieldID fid = *it;
  //     const char *field_name;
  //     runtime->retrieve_name(fspace, fid, field_name);
  //     vtkCPInputDataDescription* idd = dataDescription->GetInputDescriptionByName(field_name);
  //     if (idd->IsFieldNeeded(field_name, vtkDataObject::POINT) == true)
  //     {
  //       if (VTKGrid->GetPointData()->GetNumberOfArrays() == 0)
  //       {
  //         vtkNew<vtkDoubleArray> arr;
  //         arr->SetName(field_name);
  //         arr->SetNumberOfComponents(1);
  //         arr->SetNumberOfTuples(static_cast<vtkIdType>(domain.get_volume()));
  //         VTKGrid->GetPointData()->AddArray(arr.GetPointer());
  //       }
  //
  //       vtkDoubleArray* arr = vtkDoubleArray::SafeDownCast(VTKGrid->GetPointData()->GetArray(field_name));
  //       AccessorRO<double, 3> data_acc(data, *it);
  //       arr->SetArray(data_acc.ptr(bounds.lo), domain.get_volume(), 1);
  //     }
  //     idd->SetGrid(VTKGrid);
  //   }
  //   VTKProcessor->CoProcess(dataDescription.GetPointer());
  // }

  auto get_region_domain = [&] (PhysicalRegion& r) -> Domain {
    return runtime->get_index_space_domain(ctx, r.get_logical_region().get_index_space());
  };

  const auto is_region_exists = [] (RegionRequirement& req) -> bool {
    return req.region.exists();
  };

  const auto get_volume = [] (Legion::Rect<3>& rect) -> size_t {
    return size_t(rect.hi[0] - rect.lo[0] + 1) * size_t(rect.hi[1] - rect.lo[1] + 1) * size_t(rect.hi[2] - rect.lo[2] + 1);
  };

  Legion::Rect<3> newBounds;
  newBounds.lo[0] = is_region_exists(gReqXm) ? bounds.lo[0] - 1 : bounds.lo[0];
  newBounds.hi[0] = is_region_exists(gReqXp) ? bounds.hi[0] + 1 : bounds.hi[0];
  newBounds.lo[1] = is_region_exists(gReqYm) ? bounds.lo[1] - 1 : bounds.lo[1];
  newBounds.hi[1] = is_region_exists(gReqYp) ? bounds.hi[1] + 1 : bounds.hi[1];
  newBounds.lo[2] = is_region_exists(gReqZm) ? bounds.lo[2] - 1 : bounds.lo[2];
  newBounds.hi[2] = is_region_exists(gReqZp) ? bounds.hi[2] + 1 : bounds.hi[2];

  Legion::Rect<3> relativeBounds;
  relativeBounds.lo[0] = bounds.lo[0] - newBounds.lo[0];
  relativeBounds.hi[0] = bounds.hi[0] - newBounds.lo[0];
  relativeBounds.lo[1] = bounds.lo[1] - newBounds.lo[1];
  relativeBounds.hi[1] = bounds.hi[1] - newBounds.lo[1];
  relativeBounds.lo[2] = bounds.lo[2] - newBounds.lo[2];
  relativeBounds.hi[2] = bounds.hi[2] - newBounds.lo[2];

  size_t numVoxels = 
    size_t(newBounds.hi[0] - newBounds.lo[0] + 1) * 
    size_t(newBounds.hi[1] - newBounds.lo[1] + 1) * 
    size_t(newBounds.hi[2] - newBounds.lo[2] + 1);
  printf("rank: %d, num Voxels: %zu\n", rank, numVoxels);

  struct Vec3i { int x, y, z; };
  Vec3i dst_dim;
  dst_dim.x = newBounds.hi[0] - newBounds.lo[0] + 1;
  dst_dim.y = newBounds.hi[1] - newBounds.lo[1] + 1;
  dst_dim.z = newBounds.hi[2] - newBounds.lo[2] + 1;

  auto copy_data_3d = [&] (const double* _src, double* _dst,
			   Legion::Rect<3> src_ext, Legion::Rect<3> dst_ext,
			   Vec3i src_dim, Vec3i dst_dim)
    {
      assert(get_volume(src_ext) == get_volume(dst_ext));
      for (int z = 0; z < src_ext.hi[2] - src_ext.lo[2] + 1; ++z) {
        for (int y = 0; y < src_ext.hi[1] - src_ext.lo[1] + 1; ++y) {
          const double* src = _src + src_ext.lo[0] + size_t(y + src_ext.lo[1]) * src_dim.x + size_t(z + src_ext.lo[2]) * src_dim.x * src_dim.y;
          double* dst = _dst + dst_ext.lo[0] + size_t(y + dst_ext.lo[1]) * dst_dim.x + size_t(z + dst_ext.lo[2]) * dst_dim.x * dst_dim.y;
          size_t num_elements =  src_ext.hi[0] - src_ext.lo[0] + 1;
          assert(num_elements == dst_ext.hi[0] - dst_ext.lo[0] + 1);
          memcpy(dst, src, sizeof(double) * num_elements);
        }
      }
    };

  /* allocate new data for rendering */
  assert(dataReq.privilege_fields.size() == 1);
  std::shared_ptr<char[]> newData(new char[sizeof(double) * numVoxels]);

  int index = 0;
  for (std::set<FieldID>::iterator it = dataReq.privilege_fields.begin(); it != dataReq.privilege_fields.end(); ++it, ++index)
  {
    FieldID fid = *it;
    const char *field_name;
    runtime->retrieve_name(fspace, fid, field_name);
    printf("field name: %s\n", field_name);

    /* copy the main domain */
    {
      AccessorRO<double, 3> data_access(data, *it);
      const double* rawptr = data_access.ptr(bounds.lo);

      Legion::Rect<3> src_ext;
      src_ext.lo[0] = src_ext.lo[1] = src_ext.lo[2] = 0;
      src_ext.hi[0] = bounds.hi[0] - bounds.lo[0];
      src_ext.hi[1] = bounds.hi[1] - bounds.lo[1];
      src_ext.hi[2] = bounds.hi[2] - bounds.lo[2];
      Vec3i src_dim;
      src_dim.x = src_ext.hi[0] + 1;
      src_dim.y = src_ext.hi[1] + 1;
      src_dim.z = src_ext.hi[2] + 1;

      copy_data_3d(rawptr, (double*)newData.get() + index * numVoxels, src_ext, relativeBounds, src_dim, dst_dim);
    }

    if (!is_region_exists(gReqXm)) { printf("[render] RANK: %d X_MINUS NOT EXIST\n", rank); } else {
      Domain gDomainXm = get_region_domain(gDataXm); printf("[render] RANK: %d X_MINUS size = %lu\n", rank, gDomainXm.get_volume());
    }
    if (!is_region_exists(gReqXp)) { printf("[render] RANK: %d X_PLUS  NOT EXIST\n", rank); } else {
      Domain gDomainXp = get_region_domain(gDataXp); printf("[render] RANK: %d X_PLUS  size = %lu\n", rank, gDomainXp.get_volume());
    }
    if (!is_region_exists(gReqYm)) { printf("[render] RANK: %d Y_MINUS NOT EXIST\n", rank); } else {
      Domain gDomainYm = get_region_domain(gDataYm); printf("[render] RANK: %d Y_MINUS size = %lu\n", rank, gDomainYm.get_volume());
    }
    if (!is_region_exists(gReqYp)) { printf("[render] RANK: %d Y_PLUS  NOT EXIST\n", rank); } else {
      Domain gDomainYp = get_region_domain(gDataYp); printf("[render] RANK: %d Y_PLUS  size = %lu\n", rank, gDomainYp.get_volume());
    }
    if (!is_region_exists(gReqZm)) { printf("[render] RANK: %d Z_MINUS NOT EXIST\n", rank); } else {
      Domain gDomainZm = get_region_domain(gDataZm); printf("[render] RANK: %d Z_MINUS size = %lu\n", rank, gDomainZm.get_volume());
    }
    if (!is_region_exists(gReqZp)) { printf("[render] RANK: %d Z_PLUS  NOT EXIST\n", rank); } else {
      Domain gDomainZp = get_region_domain(gDataZp); printf("[render] RANK: %d Z_PLUS  size = %lu\n", rank, gDomainZp.get_volume());
    }

    for (int i = 0; i < numVoxels; ++i) {
      printf("%f ", ((double*)newData.get())[i]);
    } printf("\n");
  }

  /* setup renderer */
  ovr::Scene scene;
  {
    ovr::array_3d_scalar_t output = std::make_shared<ovr::Array<3>>();
    output->dims.x = dst_dim.x;
    output->dims.y = dst_dim.y;
    output->dims.z = dst_dim.z;
    output->type = ovr::VALUE_TYPE_DOUBLE;
    output->acquire_data(std::move(newData));

    ovr::scene::Volume volume;
    volume.type = ovr::scene::Volume::STRUCTURED_REGULAR_VOLUME;
    volume.structured_regular.data = output;
    volume.structured_regular.grid_origin.x = newBounds.lo[0];
    volume.structured_regular.grid_origin.y = newBounds.lo[1];
    volume.structured_regular.grid_origin.z = newBounds.lo[2];

    ovr::scene::TransferFunction tfn;
    tfn.value_range = ovr::vec2f(0.f, 20.f);
    tfn.color = ovr::CreateColorMap("diverging/RdBu");
    tfn.opacity = ovr::CreateArray1DScalar(std::vector<float>{ 0.f, 1.f });

    ovr::scene::Model model;
    model.type = ovr::scene::Model::VOLUMETRIC_MODEL;
    model.volume_model.volume = volume;
    model.volume_model.transfer_function = tfn;

    ovr::scene::Instance instance;
    instance.models.push_back(model);
    instance.transform = ovr::affine3f::translate(ovr::vec3f(0));

    scene.instances.push_back(instance);
    scene.spp = 8;
  }

  ovr::Camera cam;
  cam.from.x = camera->from[0];
  cam.from.y = camera->from[1];
  cam.from.z = camera->from[2];
  cam.up.x = camera->up[0];
  cam.up.y = camera->up[1];
  cam.up.z = camera->up[2];
  cam.at.x = camera->at[0];
  cam.at.y = camera->at[1];
  cam.at.z = camera->at[2];

  static int ac = 1;
  static const char* av = "ovr";
  static auto ren = create_renderer("ospray");
  ren->init(ac, &av, scene, cam);
  ren->set_fbsize(ovr::vec2i(gImageWidth, gImageHeight));
  ren->set_sample_per_pixel(8);
  ren->set_path_tracing(false);
  ren->commit();
  ren->render();

  ovr::MainRenderer::FrameBufferData frame;
  ren->mapframe(&frame);

  // ovr::save_image("input" + std::to_string(rank) + ".png", (ovr::vec4f*)frame.rgba->to_cpu()->data(),
  //                 /**/ gImageWidth, gImageHeight);

  float* pixels = (float*)frame.rgba->to_cpu()->data();

  /* retrieve RGBA channel and depth information */
// #if 0
//   PNGImage *pngimage = new PNGImage();
//   std::stringstream ss;
//   ss << "rank" << 0 /*rank*/ << "/RenderView1_" << *timestep << ".png";
//   read_png_file(ss.str().c_str(), pngimage);
//   ss.str(std::string());
//   ss.clear();
//   ss << "rank" << 0 /*rank*/ << "/z_buffer_" << *timestep << ".vti";
//   vtkXMLImageDataReader *reader = vtkXMLImageDataReader::New();
//   reader->SetFileName(ss.str().c_str());
//   reader->Update();
//   vtkImageData *buffer = reader->GetOutput();
//   vtkFloatArray *z_buf = vtkFloatArray::SafeDownCast(buffer->GetPointData()->GetArray(0));
// #endif

  // std::stringstream tilename;
  // // tilename << "tile-" << 7 - rank << ".dat"; 
  // tilename << "tile-" << rank << ".dat";
  // FILE* file = fopen(tilename.str().c_str(), "rb");
  // if (!file) {
  //   fprintf(stderr, "fopen('%s', 'rb') failed: %d", tilename.str().c_str(), errno);
  //   return;
  // }
  // int width, height;
  // fscanf(file, "P6\n%i %i\n255\n", &width, &height);
  // uint8_t* pixels = new uint8_t[width * height * 4];
  // for (int y = 0; y < height; y++) {
  //   unsigned char* in = &pixels[4 * (height - 1 - y) * width];
  //   fread(in, sizeof(char), 4 * width, file);
  // }
  // float depth;
  // fscanf(file, "\nDEPTH\n%f\n", &depth);
  // fclose(file);
  // std::cout << rank << " w " << width << " h " << height << " d " << depth << std::endl;
 
  /* put them together */
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
    int y = point[1];
    int x = point[0];
    a[*pir] = pixels[4 * (gImageWidth * y + x) + 3]; // pixels[4 * (gImageWidth * y + x) + 3] / 255.f;
    r[*pir] = pixels[4 * (gImageWidth * y + x) + 0]; // pixels[4 * (gImageWidth * y + x) + 0] / 255.f;// * a[*pir];
    g[*pir] = pixels[4 * (gImageWidth * y + x) + 1]; // pixels[4 * (gImageWidth * y + x) + 1] / 255.f;// * a[*pir];
    b[*pir] = pixels[4 * (gImageWidth * y + x) + 2]; // pixels[4 * (gImageWidth * y + x) + 2] / 255.f;// * a[*pir];
    z[*pir] = 0;
    u[*pir] = 0;
  }

  //  delete pngimage;
}

static int save_image_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  const SaveImageArgs *args = (const SaveImageArgs *)(task->args);
  const ImageDescriptor imageDescriptor = args->imageDescriptor;
  const char *outDir = args->outdir;
  const int timestep = args->timestep;

  PhysicalRegion image = regions[0];
  IndexSpace indexSpace = image.get_logical_region().get_index_space();
  Legion::Rect<3> bounds = runtime->get_index_space_domain(ctx, indexSpace);
  std::vector<legion_field_id_t> imageFields;
  image.get_fields(imageFields);

  AccessorRO<ImageReduction::PixelField, 3> r(image, imageFields[0]);
  AccessorRO<ImageReduction::PixelField, 3> g(image, imageFields[1]);
  AccessorRO<ImageReduction::PixelField, 3> b(image, imageFields[2]);
  AccessorRO<ImageReduction::PixelField, 3> a(image, imageFields[3]);

  int rank = runtime->find_local_MPI_rank();
  printf("RANK: %d SAVE TIMESTEP: %d\n", rank, timestep);

  char filename[1024];
  sprintf(filename, "%s/image_%d_%d_%d.%05d.png", outDir, (int)bounds.lo.x, (int)bounds.lo.y, (int)bounds.lo.z, timestep);

  IndexSpace saveIndexSpace = image.get_logical_region().get_index_space();
  Legion::Rect<3> saveRect = runtime->get_index_space_domain(ctx, saveIndexSpace);
  PointInRectIterator<3> pir(saveRect);
  ImageReduction::PixelField* BB = (ImageReduction::PixelField*)b.ptr(*pir);
  ImageReduction::PixelField* GG = (ImageReduction::PixelField*)g.ptr(*pir);
  ImageReduction::PixelField* RR = (ImageReduction::PixelField*)r.ptr(*pir);
  ImageReduction::PixelField* AA = (ImageReduction::PixelField*)a.ptr(*pir);

  // TODO scale premultiplied RGB back ...
  write_png_file(filename, imageDescriptor.width, imageDescriptor.height, RR, GG, BB, AA);
  std::cout << "rank" << rank << " wrote image " << filename << std::endl;
  return 0;
}

// Called from mapper before runtime has started
void cxx_preinitialize()
{
  ImageReduction::preinitializeBeforeRuntimeStarts();

  // Allocate physical regions contiguously in memory
  LayoutConstraintRegistrar layout_registrar(FieldSpace::NO_SPACE, "SOA layout");
  std::vector<DimensionKind> dim_order(4);
  dim_order[0] = DIM_X;
  dim_order[1] = DIM_Y;
  dim_order[2] = DIM_Z;
  dim_order[3] = DIM_F; // fields go last for SOA
  layout_registrar.add_constraint(OrderingConstraint(dim_order, true/*contig*/));
  LayoutConstraintID soa_layout_id = Runtime::preregister_layout(layout_registrar);

  // Preregister render task
  gRenderTaskID = Legion::Runtime::generate_static_task_id();
  TaskVariantRegistrar registrar(gRenderTaskID, "render_task");
  registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
    .add_layout_constraint_set(0/*index*/, soa_layout_id)
    .add_layout_constraint_set(1/*index*/, soa_layout_id)
    .add_layout_constraint_set(2/*index*/, soa_layout_id);
  Runtime::preregister_task_variant<render_task>(registrar, "render_task");

  // // Preregister render task (CUDA variant)
  // gRenderTaskID = <same id reused>
  // TaskVariantRegistrar registrar(gRenderTaskID, "render_task_cuda");
  // registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC))
  //   .add_layout_constraint_set(0/*index*/, soa_layout_id)
  //   .add_layout_constraint_set(1/*index*/, soa_layout_id)
  //   .add_layout_constraint_set(2/*index*/, soa_layout_id);
  // Runtime::preregister_task_variant<render_task>(registrar, "render_task_cuda");
  
  // Preregister save image task
  gSaveImageTaskID = Legion::Runtime::generate_static_task_id();
  TaskVariantRegistrar registrarSaveImage(gSaveImageTaskID, "save_image_task");
  registrarSaveImage.add_constraint(ProcessorConstraint(Processor::LOC_PROC)).add_layout_constraint_set(0/*index*/, soa_layout_id);
  Runtime::preregister_task_variant<int, save_image_task>(registrarSaveImage, "save_image_task");

  // Preregister projection functors (non periodic)
  // TODO in this project, the "projection_bounds" is hard-coded.
  const Rect<3> projection_bounds(
    Point<3>(0,0,0),
    Point<3>(1,1,1)
    // Point<3>(0,7,0)
  );

  Runtime::preregister_projection_functor(PROJECT_X_PLUS,  new StencilProjectionFunctor< 1,0,0,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_X_MINUS, new StencilProjectionFunctor<-1,0,0,false/*periodic*/>(projection_bounds));

  Runtime::preregister_projection_functor(PROJECT_Y_PLUS,  new StencilProjectionFunctor<0, 1,0,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_Y_MINUS, new StencilProjectionFunctor<0,-1,0,false/*periodic*/>(projection_bounds));

  Runtime::preregister_projection_functor(PROJECT_Z_PLUS,  new StencilProjectionFunctor<0,0, 1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_Z_MINUS, new StencilProjectionFunctor<0,0,-1,false/*periodic*/>(projection_bounds));

  Runtime::preregister_projection_functor(PROJECT_CPPP, new StencilProjectionFunctor< 1, 1, 1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_CPPM, new StencilProjectionFunctor< 1, 1,-1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_CPMP, new StencilProjectionFunctor< 1,-1, 1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_CPMM, new StencilProjectionFunctor< 1,-1,-1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_CMPP, new StencilProjectionFunctor<-1, 1, 1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_CMPM, new StencilProjectionFunctor<-1, 1,-1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_CMMP, new StencilProjectionFunctor<-1,-1, 1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_CMMM, new StencilProjectionFunctor<-1,-1,-1,false/*periodic*/>(projection_bounds));
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

  // if (VTKProcessor == NULL)
  // {
  //   std::stringstream ss;
  //   ss << "rank" << rank;
  //   VTKProcessor = vtkCPProcessor::New();
  //   VTKProcessor->Initialize(ss.str().c_str());
  // }
  // else
  // {
  //   VTKProcessor->RemoveAllPipelines();
  // }

  // const InputArgs &args = Runtime::get_input_args();

  // for (int i = 0; i < args.argc; i++)
  // {
  //   if (!strcmp(args.argv[i], "-pipeline"))
  //   {
  //     vtkNew<vtkCPPythonScriptPipeline> pipeline;
  //     pipeline->Initialize(args.argv[i+1]);
  //     VTKProcessor->AddPipeline(pipeline.GetPointer());
  //   }
  // }
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

  IndexTaskLauncher renderLauncher(gRenderTaskID, compositor->renderImageDomain(), TaskArgument(args, argSize), argMap, Predicate::TRUE_PRED, false, imageReductionMapperID);

  // Input data
  RegionRequirement req0(imageDescriptor.simulationLogicalPartition, 0, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  for(int i = 0; i < imageDescriptor.numPFields; ++i) {
    req0.add_field(imageDescriptor.pFields[i]);
  }
  renderLauncher.add_region_requirement(req0);

  // Output image per task
  RegionRequirement req1(compositor->renderImagePartition(), 0, WRITE_DISCARD, EXCLUSIVE, compositor->sourceImage(), imageReductionMapperID);
  req1.add_field(ImageReduction::FID_FIELD_R);
  req1.add_field(ImageReduction::FID_FIELD_G);
  req1.add_field(ImageReduction::FID_FIELD_B);
  req1.add_field(ImageReduction::FID_FIELD_A);
  req1.add_field(ImageReduction::FID_FIELD_Z);
  req1.add_field(ImageReduction::FID_FIELD_USERDATA);
  renderLauncher.add_region_requirement(req1);

  // Requirements for ghost regions
  RegionRequirement req_xm(imageDescriptor.simulationLogicalPartition, PROJECT_X_MINUS, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_xp(imageDescriptor.simulationLogicalPartition, PROJECT_X_PLUS,  READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_ym(imageDescriptor.simulationLogicalPartition, PROJECT_Y_MINUS, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_yp(imageDescriptor.simulationLogicalPartition, PROJECT_Y_PLUS,  READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_zm(imageDescriptor.simulationLogicalPartition, PROJECT_Z_MINUS, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_zp(imageDescriptor.simulationLogicalPartition, PROJECT_Z_PLUS,  READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);

  for(int i = 0; i < imageDescriptor.numPFields; ++i) {
    req_xm.add_field(imageDescriptor.pFields[i]);
    req_xp.add_field(imageDescriptor.pFields[i]);
    req_ym.add_field(imageDescriptor.pFields[i]);
    req_yp.add_field(imageDescriptor.pFields[i]);
    req_zm.add_field(imageDescriptor.pFields[i]);
    req_zp.add_field(imageDescriptor.pFields[i]);
  }

  renderLauncher.add_region_requirement(req_xm);
  renderLauncher.add_region_requirement(req_xp);
  renderLauncher.add_region_requirement(req_ym);
  renderLauncher.add_region_requirement(req_yp);
  renderLauncher.add_region_requirement(req_zm);
  renderLauncher.add_region_requirement(req_zp);

  // Finalize
  runtime->execute_index_space(ctx, renderLauncher);
}

void cxx_reduce(legion_context_t ctx_,
                Camera camera)
{
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();
  ImageReduction* compositor = gImageCompositor;

  // Cf * af + (1-af) * Cb * ab
  compositor->set_blend_func(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  compositor->set_blend_equation(GL_FUNC_ADD);

  float cameraDirection[] = {
    (float)(camera.at[0] - camera.from[0]),
    (float)(camera.at[1] - camera.from[1]),
    (float)(camera.at[2] - camera.from[2])
  };

  compositor->reduceImages(ctx, cameraDirection);
}

// save the final image to disk
void cxx_saveImage(legion_runtime_t runtime_,
                   legion_context_t ctx_,
                   const char* outDir,
                   int timestep)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  // save the image
  ImageReduction* compositor = gImageCompositor;
  ImageDescriptor imageDescriptor = compositor->imageDescriptor();

  SaveImageArgs args(imageDescriptor, strdup(outDir), timestep);

  TaskLauncher saveImageLauncher(gSaveImageTaskID, TaskArgument(&args, sizeof(args)), Predicate::TRUE_PRED, imageReductionMapperID);
  DomainPoint slice0 = Legion::Point<3>::ZEROES();
  LogicalRegion imageSlice0 = runtime->get_logical_subregion_by_color(compositor->compositeImagePartition(), slice0);
  RegionRequirement req(imageSlice0, READ_ONLY, EXCLUSIVE, compositor->sourceImage(), imageReductionMapperID);
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
                              const char* outDir,
                              int timestep)
{
  Runtime *runtime = CObjectWrapper::unwrap(runtime_);
  Context ctx = CObjectWrapper::unwrap(ctx_)->context();

  ImageReduction* compositor = gImageCompositor;
  ImageDescriptor imageDescriptor = compositor->imageDescriptor();
  ArgumentMap argMap;

  SaveImageArgs args(imageDescriptor, strdup(outDir), timestep);

  IndexTaskLauncher saveImageLauncher(gSaveImageTaskID, compositor->compositeImageDomain(), TaskArgument(&args, sizeof(args)), argMap, Predicate::TRUE_PRED, false, imageReductionMapperID);
  RegionRequirement req(compositor->compositeImagePartition(), 0, READ_ONLY, EXCLUSIVE, compositor->sourceImage(), imageReductionMapperID);
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

