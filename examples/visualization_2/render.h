//
// render.h
//


#include "legion.h"
#include "legion_c.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
  float xMin;
  float yMin;
  float zMin;
  float xMax;
  float yMax;
  float zMax;
  int numNodes[3];
} Model;

typedef struct {
  float from[3];
  float at[3];
  float up[3];
} Camera;


void cxx_preinitialize(legion_mapper_id_t mapperID);

typedef struct {
  legion_index_space_t indexSpace;
  legion_logical_region_t imageX;
  legion_index_space_t colorSpace;
  legion_logical_partition_t p_Image;
  legion_field_id_t imageFields[6];
} RegionPartition;

RegionPartition cxx_initialize(
                               legion_runtime_t runtime_,
                               legion_context_t ctx_,
                               legion_logical_partition_t partition_);

void cxx_render(legion_runtime_t runtime_,
                legion_context_t ctx_,
                legion_physical_region_t image_[],
                legion_field_id_t imageFields[],
                int numImageFields,
                legion_logical_region_t r_,
                legion_logical_partition_t p_,
                legion_field_id_t pFields[],
                int numPFields);

void cxx_reduce(legion_runtime_t runtime_,
                  legion_context_t ctx_,
                  const char* outDir
                  );



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

