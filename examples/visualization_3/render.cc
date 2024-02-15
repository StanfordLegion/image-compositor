//
// render.cc
//
// example C++ interface to support visualization
//

#include "image.h"
#include "render.h"
#include "legion_visualization.h"
#include "image_reduction_mapper.h"

#include <sstream>
#include <cstdio>

#include "s3d_projection.h"

#include <ovr/common/imageio.h>
#include <ovr/renderer.h>
#include <ovr/serializer/serializer.h>

#define _T {std::cout<<__FILE__<<" "<<__LINE__<<" "<<__FUNCTION__<<std::endl;}

using ImageReduction = Legion::Visualization::ImageReduction;
using ImageDescriptor = Legion::Visualization::ImageDescriptor;
using namespace Legion;

// global data
Legion::MapperID imageReductionMapperID = 1;
int gRenderTaskID = 0;
static ImageReduction* gImageCompositor = nullptr;
static int gSaveImageTaskID = 0;
static int gImageWidth  = 800;
static int gImageHeight = 600;

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

struct Vec3i { int x, y, z; };

void legion_wait_on_mpi()
{
  handshake.legion_wait_on_mpi();
}

void legion_handoff_to_mpi()
{
  handshake.legion_handoff_to_mpi();
}

auto IsRegionExists(const RegionRequirement& req) -> bool 
{
  return req.region.exists();
}

auto GetSize(const Legion::Rect<3>& rect) -> size_t 
{
  return size_t(rect.hi[0] - rect.lo[0] + 1) * size_t(rect.hi[1] - rect.lo[1] + 1) * size_t(rect.hi[2] - rect.lo[2] + 1);
}

auto GetDimensions(const Legion::Rect<3>& rect) -> Vec3i 
{
  Vec3i strides;
  strides.x = rect.hi[0] - rect.lo[0] + 1;
  strides.y = rect.hi[1] - rect.lo[1] + 1;
  strides.z = rect.hi[2] - rect.lo[2] + 1;
  return strides;
}

auto CopyData3D(const double* _src, double* _dst, Legion::Rect<3> src_ext, Legion::Rect<3> dst_ext, Vec3i src_dim, Vec3i dst_dim)
{
  assert(GetSize(src_ext) == GetSize(dst_ext));
  for (int z = 0; z < src_ext.hi[2] - src_ext.lo[2] + 1; ++z) {
    for (int y = 0; y < src_ext.hi[1] - src_ext.lo[1] + 1; ++y) {
      const double* src = _src + src_ext.lo[0] + size_t(y + src_ext.lo[1]) * src_dim.x + size_t(z + src_ext.lo[2]) * src_dim.x * src_dim.y;
      double* dst = _dst + dst_ext.lo[0] + size_t(y + dst_ext.lo[1]) * dst_dim.x + size_t(z + dst_ext.lo[2]) * dst_dim.x * dst_dim.y;
      const size_t num_elements = src_ext.hi[0] - src_ext.lo[0] + 1;
      assert(num_elements == dst_ext.hi[0] - dst_ext.lo[0] + 1);
      memcpy(dst, src, sizeof(double) * num_elements);
    }
  }
}

#ifdef OVR_BUILD_OPTIX7

__global__
void copy_framebuffer_cuda(const int width, const int height, float* framebuffer, 
                           ImageReduction::PixelField* r,
                           ImageReduction::PixelField* g,
                           ImageReduction::PixelField* b,
                           ImageReduction::PixelField* a,
                           ImageReduction::PixelField* z,
                           ImageReduction::PixelField* u)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width*height)
  {
    r[i] = framebuffer[4 * i + 0]; 
    g[i] = framebuffer[4 * i + 1]; 
    b[i] = framebuffer[4 * i + 2]; 
    a[i] = framebuffer[4 * i + 3]; 
    z[i] = 0.f;
    u[i] = 0.f;
  }
}

void
__global__ print(const double* data)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  printf("value = %f\n", data[i]);
}

#endif

static ovr::MainRenderer::FrameBufferData
render_task_common(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime, 
                   std::shared_ptr<ovr::MainRenderer> renderer)
{
  const int rank = runtime->find_local_MPI_rank();

  PhysicalRegion data  = regions[0];

  IndexSpace indexSpace = data.get_logical_region().get_index_space();
  Domain domain = runtime->get_index_space_domain(ctx, indexSpace);
  Legion::Rect<3> bounds = domain;

  LogicalRegion dataRegion = data.get_logical_region();
  FieldSpace fspace = dataRegion.get_field_space();
  RegionRequirement dataReq = task->regions[0];

  // Access ghost regions
  RegionRequirement gReqXm = task->regions[2];
  RegionRequirement gReqYm = task->regions[3];
  RegionRequirement gReqZm = task->regions[4];
  RegionRequirement gReqCm = task->regions[5];
  RegionRequirement gReqZ0 = task->regions[6];
  RegionRequirement gReqY0 = task->regions[7];
  RegionRequirement gReqX0 = task->regions[8];

  PhysicalRegion gDataXm = regions[2];
  PhysicalRegion gDataYm = regions[3];
  PhysicalRegion gDataZm = regions[4];
  PhysicalRegion gDataCm = regions[5];
  PhysicalRegion gDataZ0 = regions[6];
  PhysicalRegion gDataY0 = regions[7];
  PhysicalRegion gDataX0 = regions[8];

  // Access camera information
  char* argsPtr = (char*)task->args;
  ImageDescriptor* imageDescriptor = (ImageDescriptor*)(argsPtr);
  Camera* camera = (Camera*)(argsPtr + sizeof(ImageDescriptor));
  int* timestep = (int*)(argsPtr + sizeof(ImageDescriptor) + sizeof(Camera));

  const auto GetRegionDomain = [&] (const PhysicalRegion& r) -> Domain {
    return runtime->get_index_space_domain(ctx, r.get_logical_region().get_index_space());
  };

  Legion::Rect<3> boundsWithGhosts;
  boundsWithGhosts.lo[0] = IsRegionExists(gReqXm) ? bounds.lo[0] - 1 : bounds.lo[0];
  boundsWithGhosts.lo[1] = IsRegionExists(gReqYm) ? bounds.lo[1] - 1 : bounds.lo[1];
  boundsWithGhosts.lo[2] = IsRegionExists(gReqZm) ? bounds.lo[2] - 1 : bounds.lo[2];
  boundsWithGhosts.hi[0] = /*IsRegionExists(gReqXp) ? bounds.hi[0] + 0 :*/ bounds.hi[0];
  boundsWithGhosts.hi[1] = /*IsRegionExists(gReqYp) ? bounds.hi[1] + 0 :*/ bounds.hi[1];
  boundsWithGhosts.hi[2] = /*IsRegionExists(gReqZp) ? bounds.hi[2] + 0 :*/ bounds.hi[2];
  const Vec3i dimsWithGhosts = GetDimensions(boundsWithGhosts);

  const size_t totalNumVoxels = GetSize(boundsWithGhosts); // printf("rank: %d, num Voxels: %zu\n", rank, totalNumVoxels);

  // allocate new data for rendering 
  std::shared_ptr<char[]> volumeWithGhost(new char[sizeof(double) * totalNumVoxels]);
#ifdef OVR_BUILD_OPTIX7
  double* volumePtr = (double*)volumeWithGhost.get();
#else
  double* volumePtr = (double*)volumeWithGhost.get();
#endif

  const auto CopyGhostLo = [GetRegionDomain, dimsWithGhosts, boundsWithGhosts] (std::set<FieldID>::iterator& it, double* dst, int axis, PhysicalRegion& ghost_data)
  {
    const Domain ghost_domain = GetRegionDomain(ghost_data);
    const Legion::Rect<3> ghost_bounds = ghost_domain;

    AccessorRO<double, 3> dataAccess(ghost_data, *it);
    const double* _rawptr = dataAccess.ptr(ghost_bounds.lo); // how dows this work? what is the data layout?

#ifdef OVR_BUILD_OPTIX7
    size_t nn = GetSize(ghost_bounds);
    std::vector<double> tmp(nn);
    cudaMemcpy(tmp.data(), _rawptr, nn*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    const double* rawptr = tmp.data();
#else
    const double* rawptr = _rawptr;
#endif

    Legion::Rect<3> src_ext;
    src_ext.hi[axis] = src_ext.lo[axis] = ghost_bounds.hi[axis] - ghost_bounds.lo[axis];
    for (int a = 0; a < 3; ++a) {
      if (a == axis) continue;
      src_ext.lo[a] = 0;
      src_ext.hi[a] = ghost_bounds.hi[a] - ghost_bounds.lo[a];
    }

    const Vec3i src_dim = GetDimensions(ghost_bounds);

    Legion::Rect<3> dst_ext;
    dst_ext.lo[axis] = ghost_bounds.hi[axis] - boundsWithGhosts.lo[axis];
    dst_ext.hi[axis] = ghost_bounds.hi[axis] - boundsWithGhosts.lo[axis];
    for (int a = 0; a < 3; ++a) {
      if (a == axis) continue;
      dst_ext.lo[a] = ghost_bounds.lo[a] - boundsWithGhosts.lo[a];
      dst_ext.hi[a] = ghost_bounds.hi[a] - boundsWithGhosts.lo[a];
    }

    CopyData3D(rawptr, dst, src_ext, dst_ext, src_dim, dimsWithGhosts);
  };

  const auto CopyGhostMiddle = [GetRegionDomain, dimsWithGhosts, boundsWithGhosts] (std::set<FieldID>::iterator& it, double* dst, int axis, PhysicalRegion& ghost_data)
  {
    const Domain ghost_domain = GetRegionDomain(ghost_data);
    const Legion::Rect<3> ghost_bounds = ghost_domain;

    AccessorRO<double, 3> dataAccess(ghost_data, *it);
    const double* _rawptr = dataAccess.ptr(ghost_bounds.lo); // how dows this work? what is the data layout?

#ifdef OVR_BUILD_OPTIX7
    size_t nn = GetSize(ghost_bounds);
    std::vector<double> tmp(nn);
    cudaMemcpy(tmp.data(), _rawptr, nn*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    const double* rawptr = tmp.data();
#else
    const double* rawptr = _rawptr;
#endif

    Legion::Rect<3> src_ext;
    src_ext.lo[axis] = 0;
    src_ext.hi[axis] = ghost_bounds.hi[axis] - ghost_bounds.lo[axis];
    for (int a = 0; a < 3; ++a) {
      if (a == axis) continue;
      src_ext.hi[a] = src_ext.lo[a] = ghost_bounds.hi[a] - ghost_bounds.lo[a];
    }

    const Vec3i src_dim = GetDimensions(ghost_bounds);

    Legion::Rect<3> dst_ext;
    dst_ext.lo[axis] = ghost_bounds.lo[axis] - boundsWithGhosts.lo[axis];
    dst_ext.hi[axis] = ghost_bounds.hi[axis] - boundsWithGhosts.lo[axis];
    for (int a = 0; a < 3; ++a) {
      if (a == axis) continue;
      dst_ext.lo[a] = dst_ext.hi[a] = ghost_bounds.hi[a] - boundsWithGhosts.lo[a];
    }

    CopyData3D(rawptr, dst, src_ext, dst_ext, src_dim, dimsWithGhosts);
  };

  assert(dataReq.privilege_fields.size() == 1);
  int index = 0;
  for (std::set<FieldID>::iterator it = dataReq.privilege_fields.begin(); it != dataReq.privilege_fields.end(); ++it, ++index)
  {
    FieldID fid = *it;

    /* copy the main domain */
    {
      AccessorRO<double, 3> dataAccess(data, *it);
      const double* _rawptr = dataAccess.ptr(bounds.lo);

#ifdef OVR_BUILD_OPTIX7
      size_t nn = GetSize(ghost_bounds);
      std::vector<double> tmp(nn);
      cudaMemcpy(tmp.data(), _rawptr, nn*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      const double* rawptr = tmp.data();
#else
      const double* rawptr = _rawptr;
#endif

      Legion::Rect<3> srcExt;
      srcExt.lo[0] = srcExt.lo[1] = srcExt.lo[2] = 0;
      srcExt.hi[0] = bounds.hi[0] - bounds.lo[0];
      srcExt.hi[1] = bounds.hi[1] - bounds.lo[1];
      srcExt.hi[2] = bounds.hi[2] - bounds.lo[2];
      const Vec3i srcDim = GetDimensions(bounds);

      Legion::Rect<3> dstExt;
      dstExt.lo[0] = bounds.lo[0] - boundsWithGhosts.lo[0];
      dstExt.hi[0] = bounds.hi[0] - boundsWithGhosts.lo[0];
      dstExt.lo[1] = bounds.lo[1] - boundsWithGhosts.lo[1];
      dstExt.hi[1] = bounds.hi[1] - boundsWithGhosts.lo[1];
      dstExt.lo[2] = bounds.lo[2] - boundsWithGhosts.lo[2];
      dstExt.hi[2] = bounds.hi[2] - boundsWithGhosts.lo[2];

      CopyData3D(rawptr, volumePtr + index * totalNumVoxels, srcExt, dstExt, srcDim, dimsWithGhosts);
    }

    if (IsRegionExists(gReqXm)) CopyGhostLo(it, volumePtr + index * totalNumVoxels, 0, gDataXm);
    if (IsRegionExists(gReqYm)) CopyGhostLo(it, volumePtr + index * totalNumVoxels, 1, gDataYm);
    if (IsRegionExists(gReqZm)) CopyGhostLo(it, volumePtr + index * totalNumVoxels, 2, gDataZm);
    
    if (IsRegionExists(gReqZ0)) CopyGhostMiddle(it, volumePtr + index * totalNumVoxels, 2, gDataZ0);
    if (IsRegionExists(gReqY0)) CopyGhostMiddle(it, volumePtr + index * totalNumVoxels, 1, gDataY0);
    if (IsRegionExists(gReqX0)) CopyGhostMiddle(it, volumePtr + index * totalNumVoxels, 0, gDataX0);
    
    if (IsRegionExists(gReqCm)) {
      const Domain ghost_domain = GetRegionDomain(gDataCm);
      const Legion::Rect<3> ghost_bounds = ghost_domain;
    
      AccessorRO<double, 3> dataAccess(gDataCm, *it);
      const double* _rawptr = dataAccess.ptr(ghost_bounds.lo); // how dows this work? what is the data layout?

#ifdef OVR_BUILD_OPTIX7
      size_t nn = GetSize(ghost_bounds);
      std::vector<double> tmp(nn);
      cudaMemcpy(tmp.data(), _rawptr, nn*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      const double* rawptr = tmp.data();
#else
      const double* rawptr = _rawptr;
#endif

      Legion::Rect<3> src_ext;
      src_ext.hi[0] = src_ext.lo[0] = ghost_bounds.hi[0] - ghost_bounds.lo[0];
      src_ext.hi[1] = src_ext.lo[1] = ghost_bounds.hi[1] - ghost_bounds.lo[1];
      src_ext.hi[2] = src_ext.lo[2] = ghost_bounds.hi[2] - ghost_bounds.lo[2];
    
      const Vec3i src_dim = GetDimensions(ghost_bounds);
    
      Legion::Rect<3> dst_ext;
      dst_ext.lo[0] = dst_ext.hi[0] = ghost_bounds.hi[0] - boundsWithGhosts.lo[0];
      dst_ext.lo[1] = dst_ext.hi[1] = ghost_bounds.hi[1] - boundsWithGhosts.lo[1];
      dst_ext.lo[2] = dst_ext.hi[2] = ghost_bounds.hi[2] - boundsWithGhosts.lo[2];
    
      CopyData3D(rawptr, volumePtr + index * totalNumVoxels, src_ext, dst_ext, src_dim, dimsWithGhosts);
    }
  }

  /* setup renderer */
  ovr::Scene scene;
  {
    ovr::array_3d_scalar_t output = std::make_shared<ovr::Array<3>>();
    output->dims.x = dimsWithGhosts.x;
    output->dims.y = dimsWithGhosts.y;
    output->dims.z = dimsWithGhosts.z;
    output->type = ovr::VALUE_TYPE_DOUBLE;

    // std::ofstream fw("rank" + std::to_string(rank)+ "data.txt", std::ofstream::out);
    // for (int z = 0; z < dimsWithGhosts.z; ++z) {
    //   for (int y = 0; y < dimsWithGhosts.y; ++y) {
    //     for (int x = 0; x < dimsWithGhosts.x; ++x) {
    //       fw << volumePtr[x + dimsWithGhosts.x * y + dimsWithGhosts.x * dimsWithGhosts.y * z] << "\n";
    //     }
    //   }
    // }
    // fw.close();

    output->acquire_data(std::move(volumeWithGhost));

    ovr::scene::Volume volume;
    volume.type = ovr::scene::Volume::STRUCTURED_REGULAR_VOLUME;
    volume.structured_regular.data = output;
    volume.structured_regular.grid_origin.x = boundsWithGhosts.lo[0];
    volume.structured_regular.grid_origin.y = boundsWithGhosts.lo[1];
    volume.structured_regular.grid_origin.z = boundsWithGhosts.lo[2];

    std::cout << rank << " bounds " << bounds.lo[0] << " " << bounds.lo[1] << " " << bounds.lo[2] << std::endl
                      << "        " << bounds.hi[0] << " " << bounds.hi[1] << " " << bounds.hi[2] << std::endl;
    std::cout << rank << " dimsWithGhosts " << dimsWithGhosts.x << " " << dimsWithGhosts.y << " " << dimsWithGhosts.z << std::endl;
    std::cout << rank << " grid_origin " << volume.structured_regular.grid_origin.x << " " << volume.structured_regular.grid_origin.y << " " << volume.structured_regular.grid_origin.z << std::endl;

    std::vector<float> alphas;
    for (int i = 0; i <= 1024; ++i) {
      alphas.push_back(i / 1024.f);
    }

    ovr::scene::TransferFunction tfn;
    tfn.value_range = ovr::vec2f(0.f, 1.f);
    tfn.color = ovr::CreateColorMap("diverging/RdBu");
    tfn.opacity = ovr::CreateArray1DScalar(alphas);

    // ovr::scene::TransferFunction tfn = create_scene_tfn_vidi3d("/lustre/scratch/vsyamaj/legion_s3d_viz/ptj_temp.json");
    // tfn.value_range = ovr::vec2f(0.f, 1.f);

    ovr::scene::Model model;
    model.type = ovr::scene::Model::VOLUMETRIC_MODEL;
    model.volume_model.volume = volume;
    model.volume_model.transfer_function = tfn;

    ovr::scene::Instance instance;
    instance.models.push_back(model);
    instance.transform = ovr::affine3f::translate(ovr::vec3f(0));

    scene.instances.push_back(instance);

    scene.spp = 1;
    scene.volume_sampling_rate = 100.f;
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

  scene.camera = cam;
  scene.ao_samples = 0;

  // render
  ovr::MainRenderer::FrameBufferData frame;
  {
    static const char* argv = "ovr";
    renderer->set_path_tracing(false);
    renderer->set_frame_accumulation(false); // frame accumulation gives artifacts
    renderer->set_fbsize(ovr::vec2i(gImageWidth, gImageHeight));
    renderer->init(1, &argv, scene, scene.camera);
    renderer->commit();
    renderer->render();
    renderer->mapframe(&frame);
  }

  return frame;
}

#ifdef OVR_BUILD_OPTIX7

static void render_task_gpu(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
{
  static auto ren = create_renderer("optix7");

  ovr::MainRenderer::FrameBufferData frame = render_task_common(task, regions, ctx, runtime, ren);
  float* pixels = (float*)frame.rgba->to_cuda()->data();

  PhysicalRegion image = regions[1];

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

  float* aptr = a.ptr(saveRect.lo); // device pointers
  float* rptr = r.ptr(saveRect.lo);
  float* gptr = g.ptr(saveRect.lo);
  float* bptr = b.ptr(saveRect.lo);
  float* zptr = z.ptr(saveRect.lo);
  float* uptr = u.ptr(saveRect.lo);
  const int threads_per_block = 256;
  const int num_blocks = (gImageWidth*gImageHeight + (threads_per_block-1)) / threads_per_block;
  copy_framebuffer_cuda<<<num_blocks,threads_per_block>>>(gImageWidth, gImageHeight, pixels, rptr, gptr, bptr, aptr, zptr, uptr);
}

#endif

static void render_task_cpu(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
{
  static auto ren = create_renderer("ospray");

  ovr::MainRenderer::FrameBufferData frame = render_task_common(task, regions, ctx, runtime, ren);
  float* pixels = (float*)frame.rgba->to_cpu()->data();

  PhysicalRegion image = regions[1];

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
    a[*pir] = pixels[4 * (gImageWidth * y + x) + 3]; 
    r[*pir] = pixels[4 * (gImageWidth * y + x) + 0]; 
    g[*pir] = pixels[4 * (gImageWidth * y + x) + 1]; 
    b[*pir] = pixels[4 * (gImageWidth * y + x) + 2]; 
    z[*pir] = 0;
    u[*pir] = 0;
  }
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
  // printf("RANK: %d SAVE TIMESTEP: %d\n", rank, timestep);

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
#ifdef OVR_BUILD_OPTIX7
  {
    TaskVariantRegistrar reg_gpu(gRenderTaskID, "render_task");
    reg_gpu.add_constraint(ProcessorConstraint(Processor::TOC_PROC))
      .add_layout_constraint_set(0/*index*/, soa_layout_id)
      .add_layout_constraint_set(1/*index*/, soa_layout_id)
      .add_layout_constraint_set(2/*index*/, soa_layout_id);
    Runtime::preregister_task_variant<render_task_gpu>(reg_gpu, "render_task");
  }
#endif
  {
    TaskVariantRegistrar reg_cpu(gRenderTaskID, "render_task");
    reg_cpu.add_constraint(ProcessorConstraint(Processor::LOC_PROC))
      .add_layout_constraint_set(0/*index*/, soa_layout_id)
      .add_layout_constraint_set(1/*index*/, soa_layout_id)
      .add_layout_constraint_set(2/*index*/, soa_layout_id);
    Runtime::preregister_task_variant<render_task_cpu>(reg_cpu, "render_task");
  }

  // Preregister save image task
  gSaveImageTaskID = Legion::Runtime::generate_static_task_id();
  TaskVariantRegistrar registrarSaveImage(gSaveImageTaskID, "save_image_task");
  registrarSaveImage.add_constraint(ProcessorConstraint(Processor::LOC_PROC)).add_layout_constraint_set(0/*index*/, soa_layout_id);
  Runtime::preregister_task_variant<int, save_image_task>(registrarSaveImage, "save_image_task");

  // Preregister projection functors (non periodic)
  // TODO in this project, the "projection_bounds" is hard-coded.
  const Rect<3> projection_bounds(
    Point<3>(0,0,0),
    // Point<3>(1,1,1)
    // Point<3>(0,7,0)
    // Point<3>(7,0,0)
    Point<3>(1,1,0)
    // Point<3>(3,0,0)
    // Point<3>(0,0,0)
  );

  Runtime::preregister_projection_functor(PROJECT_X_MINUS, new StencilProjectionFunctor<0,false/*plus*/,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_Y_MINUS, new StencilProjectionFunctor<1,false/*plus*/,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_Z_MINUS, new StencilProjectionFunctor<2,false/*plus*/,false/*periodic*/>(projection_bounds));

  Runtime::preregister_projection_functor(PROJECT_XM_YM_ZM,
    new StencilProjectionFunctorXYZ<-1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_XY_MINUS_Z0,
    new StencilProjectionFunctorXYZ< 2,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_XZ_MINUS_Y0,
    new StencilProjectionFunctorXYZ< 1,false/*periodic*/>(projection_bounds));
  Runtime::preregister_projection_functor(PROJECT_YZ_MINUS_X0,
    new StencilProjectionFunctorXYZ< 0,false/*periodic*/>(projection_bounds));
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
  RegionRequirement req_ym(imageDescriptor.simulationLogicalPartition, PROJECT_Y_MINUS, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_zm(imageDescriptor.simulationLogicalPartition, PROJECT_Z_MINUS, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_cm(imageDescriptor.simulationLogicalPartition, PROJECT_XM_YM_ZM, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_cz0(imageDescriptor.simulationLogicalPartition, PROJECT_XY_MINUS_Z0, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_cy0(imageDescriptor.simulationLogicalPartition, PROJECT_XZ_MINUS_Y0, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);
  RegionRequirement req_cx0(imageDescriptor.simulationLogicalPartition, PROJECT_YZ_MINUS_X0, READ_ONLY, EXCLUSIVE, imageDescriptor.simulationLogicalRegion, imageReductionMapperID);

  for(int i = 0; i < imageDescriptor.numPFields; ++i) {
    req_xm.add_field(imageDescriptor.pFields[i]);
    req_ym.add_field(imageDescriptor.pFields[i]);
    req_zm.add_field(imageDescriptor.pFields[i]);
    req_cm.add_field(imageDescriptor.pFields[i]);
    req_cz0.add_field(imageDescriptor.pFields[i]);
    req_cy0.add_field(imageDescriptor.pFields[i]);
    req_cx0.add_field(imageDescriptor.pFields[i]);
  }

  renderLauncher.add_region_requirement(req_xm);
  renderLauncher.add_region_requirement(req_ym);
  renderLauncher.add_region_requirement(req_zm);
  renderLauncher.add_region_requirement(req_cm);
  renderLauncher.add_region_requirement(req_cz0);
  renderLauncher.add_region_requirement(req_cy0);
  renderLauncher.add_region_requirement(req_cx0);

  // Finalize
  runtime->execute_index_space(ctx, renderLauncher);
}

void cxx_reduce(legion_context_t ctx_, Camera camera)
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
  // compositor->reduceImagesOrthographic(ctx, cameraDirection);
  compositor->reduceImagesPerspective(ctx, camera.from);
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
void cxx_saveIndividualImages(legion_runtime_t runtime_, legion_context_t ctx_, const char* outDir, int timestep)
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

