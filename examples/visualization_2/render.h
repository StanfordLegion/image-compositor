// render.h

#include "legion.h"
#include "legion_c.h"


#ifdef __cplusplus
extern "C" {
#endif

  typedef struct { float from[3]; float at[3]; float up[3]; } Camera;

  void cxx_preinitialize();

  void cxx_initialize( legion_runtime_t runtime_, legion_context_t ctx_,
  legion_logical_region_t region, legion_logical_partition_t partition_,
  legion_field_id_t pFields[], int numPFields);

  void cxx_render(legion_runtime_t runtime_, legion_context_t ctx_,
   Camera camera);

  void cxx_reduce(legion_context_t ctx_, float cameraDirection[3]);

  void cxx_saveImage(legion_runtime_t runtime_, legion_context_t ctx_, const
  char* outDir );

  // set this to 1 when debugging the renderer, 0 otherwise #define
  // DEBUG_INDIVIDUAL_IMAGES 1

  void cxx_saveIndividualImages(legion_runtime_t runtime_, legion_context_t
  ctx_, const char* outDir );

  void cxx_terminate();

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
template<typename FT, int N, typename T = long long>
using AccessorRO = Legion::FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = long long>
using AccessorWO = Legion::FieldAccessor<WRITE_DISCARD,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = long long>
using AccessorRW = Legion::FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
#endif
