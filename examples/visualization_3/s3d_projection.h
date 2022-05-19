
#ifndef __S3D_PROJECTION_H__
#define __S3D_PROJECTION_H__

//#include "s3d_types.h"
#include <sys/utsname.h>

enum { PCLR_XYZLO = 0, PCLR_XYZHI = 1 };
enum { PID_X = 0, PID_Y = 1, PID_Z = 2, PID_SUBRANK = 3 };

// Enumeration for Projection IDs
enum {
  // 0 is a reserved projection ID
  // PROJECT_X_PLUS = 1,
  PROJECT_X_MINUS = 2,
  // PROJECT_Y_PLUS = 3,
  PROJECT_Y_MINUS = 4,
  // PROJECT_Z_PLUS = 5,
  PROJECT_Z_MINUS = 6,

  PROJECT_XM_YM_ZM = 7,
  PROJECT_XY_MINUS_Z0 = 8,
  PROJECT_XZ_MINUS_Y0 = 9,
  PROJECT_YZ_MINUS_X0 = 10,
};

template<int DIM, bool PLUS, bool PERIODIC>
class StencilProjectionFunctor : public Legion::ProjectionFunctor {  
  template<int D>
  using LogicalRegion = Legion::LogicalRegionT<D,long long>;

  template<int D>
  using LogicalPartition = Legion::LogicalPartitionT<D,long long>;

  template<int D>
  using Rect = Legion::Rect<D,long long>;

public:
  StencilProjectionFunctor(const Rect<3> b) : bounds(b) { }

public:
  virtual Legion::LogicalRegion project(Legion::LogicalRegion upper_bound, const Legion::DomainPoint &point, const Legion::Domain &launch_domain)
  {
    assert(false); // Should never be called
    return Legion::LogicalRegion::NO_REGION;
  }

  virtual Legion::LogicalRegion project(Legion::LogicalPartition upper_bound, const Legion::DomainPoint &point, const Legion::Domain &launch_domain)
  {
    Legion::Point<3> p = point;

    // {
    //   struct utsname buf;
    //   uname(&buf);
    //   Legion::Point<3> dp = point;
    //   printf("PRE PROJECT: %d PLUS: %d NODE: %s COLOR: %lld %lld %lld POINT: %lld %lld %lld\n",
    //          DIM, PLUS, buf.nodename,
    //          dp[0], dp[1], dp[2],
    //          p[0], p[1], p[2]);
    // }

    if (PERIODIC) {
      // If the low bounds is the same as the 
      // high bound we don't want wrap-around
      if (bounds.lo[DIM] == bounds.hi[DIM])
        return Legion::LogicalRegion::NO_REGION;
      if (PLUS) {
        if (p[DIM] == bounds.hi[DIM])
        {
          p[DIM] = bounds.lo[DIM]; // wrap-around for periodic
        }
        else
        {
          p[DIM] += 1;
        }
      } else {
        if (p[DIM] == bounds.lo[DIM])
        {
          p[DIM] = bounds.hi[DIM]; // wrap-around for periodic
        }
        else
        {
          p[DIM] -= 1;
        }
      }
    } else {
      if (PLUS) {
        if (p[DIM] == bounds.hi[DIM])
          return Legion::LogicalRegion::NO_REGION; // hit the boundary
        else
          p[DIM] += 1;
      } else {
        if (p[DIM] == bounds.lo[DIM])
          return Legion::LogicalRegion::NO_REGION; // hit the boundary
        else
          p[DIM] -= 1;
      }
    }

    // {
    //   struct utsname buf;
    //   uname(&buf);
    //   Legion::Point<3> dp = point;
    //   printf("PROJECT: %d PLUS: %d NODE: %s COLOR: %lld %lld %lld POINT: %lld %lld %lld\n",
    //          DIM, PLUS, buf.nodename,
    //          dp[0], dp[1], dp[2],
    //          p[0], p[1], p[2]);
    // }

    LogicalRegion<3> rank = runtime->get_logical_subregion_by_color(LogicalPartition<3>(upper_bound), p);
    return rank;

    // LogicalPartition<3> part = runtime->get_logical_partition_by_color(rank, PID_X + DIM);
    // return runtime->get_logical_subregion_by_color(part, Legion::Point<1>(PLUS ? PCLR_XYZLO : PCLR_XYZHI));
  }
public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return /*depth change to 0*/ 0; }
public:
  const Rect<3> bounds; // global domain
};

template<int DIM, bool PERIODIC>
class StencilProjectionFunctorXYZ : public Legion::ProjectionFunctor {  
  template<int D>
  using LogicalRegion = Legion::LogicalRegionT<D,long long>;

  template<int D>
  using LogicalPartition = Legion::LogicalPartitionT<D,long long>;

  template<int D>
  using Rect = Legion::Rect<D,long long>;

public:
  StencilProjectionFunctorXYZ(const Rect<3> b) : bounds(b) { }

public:
  virtual Legion::LogicalRegion project(Legion::LogicalRegion upper_bound, const Legion::DomainPoint &point, const Legion::Domain &launch_domain)
  {
    assert(false); // Should never be called
    return Legion::LogicalRegion::NO_REGION;
  }

  virtual Legion::LogicalRegion project(Legion::LogicalPartition upper_bound, const Legion::DomainPoint &point, const Legion::Domain &launch_domain)
  {
    Legion::Point<3> p = point;

    if (PERIODIC) {

      // If the low bounds is the same as the 
      // high bound we don't want wrap-around
      if (bounds.lo[0] == bounds.hi[0]) return Legion::LogicalRegion::NO_REGION;
      if (bounds.lo[1] == bounds.hi[1]) return Legion::LogicalRegion::NO_REGION;
      if (bounds.lo[2] == bounds.hi[2]) return Legion::LogicalRegion::NO_REGION;

      if (DIM != 0) { // X Minus 
        if (p[0] == bounds.lo[0]) p[0] = bounds.hi[0]; // wrap-around for periodic
        else p[0] -= 1;
      }

      if (DIM != 1) { // Y Minus
        if (p[1] == bounds.lo[0]) p[1] = bounds.hi[1]; // wrap-around for periodic
        else p[1] -= 1;
      }

      if (DIM != 2) { // Z Minus 
        if (p[2] == bounds.hi[0]) p[2] = bounds.lo[2]; // wrap-around for periodic
        else p[2] += 1;
      }

    }
    else {
    
      if (DIM != 0) { // X Minus
        if (p[0] == bounds.lo[0]) return Legion::LogicalRegion::NO_REGION; // hit the boundary
        else p[0] -= 1;
      }

      if (DIM != 1) { // Y Minus
        if (p[1] == bounds.lo[1]) return Legion::LogicalRegion::NO_REGION; // hit the boundary
        else p[1] -= 1;
      }

      if (DIM != 2) { // Z Minus 
        if (p[2] == bounds.lo[2]) return Legion::LogicalRegion::NO_REGION; // hit the boundary
        else p[2] -= 1;
      }

    }

    // {
    //   struct utsname buf;
    //   uname(&buf);
    //   Legion::Point<3> dp = point;
    //   printf("DIM: %d NODE: %s COLOR: %lld %lld %lld POINT: %lld %lld %lld\n",
    //          DIM, buf.nodename, dp[0], dp[1], dp[2], p[0], p[1], p[2]);
    // }

    LogicalRegion<3> rank = runtime->get_logical_subregion_by_color(LogicalPartition<3>(upper_bound), p);
    return rank;
  }
public:
  virtual bool is_functional(void) const { return true; }
  virtual bool is_exclusive(void) const { return true; }
  virtual unsigned get_depth(void) const { return /*depth change to 0*/ 0; }
public:
  const Rect<3> bounds; // global domain
};

#endif // __S3D_PROJECTION_H__
