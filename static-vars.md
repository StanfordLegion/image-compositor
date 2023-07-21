For static variables defined in File [image_reduction.h](include/image_reduction.h):
```c++
namespace Legion::Visualization::ImageReduction 
{
  static int mNodeID; /* not used at all */ 
  /* cannot be static */
  static std::vector<CompositeProjectionFunctor*> *mCompositeProjectionFunctor;
  static std::vector<Domain> *mHierarchicalTreeDomain;
  /* safe, compile time constant */ 
  static const int numMatrixElements4x4 = 16;
  /* for display task, which is not used currently */
  static GLfloat mGlViewTransform[numMatrixElements4x4];
  /* safe, looks like just a constant color, should be the same for all instances */ 
  static PixelField mGlConstantColor[numPixelFields];
  /* These 3 variables can remain constant if only alpha blending is used. 
     For (sort-last) visualization, only alpha blending is used. */
  static GLenum mGlBlendEquation;
  static GLenum mGlBlendFunctionSource;
  static GLenum mGlBlendFunctionDestination;
  /* cannot be static */
  static TaskID mCompositeTaskID;
  /* for display task, which is not used */
  static TaskID mDisplayTaskID; 
  /* cannot be static */
  static KDTree<image_region_dimensions, long long int>* mSimulationKDTree;
  static KDTree<image_region_dimensions, long long int>* mImageKDTree;
}
```

<span style="background-color: #FF0000">So the variables need to be changed are:</span>
```c++
static std::vector<CompositeProjectionFunctor*> *mCompositeProjectionFunctor;
static std::vector<Domain> *mHierarchicalTreeDomain;
static TaskID mCompositeTaskID;
static KDTree<image_region_dimensions, long long int>* mSimulationKDTree;
static KDTree<image_region_dimensions, long long int>* mImageKDTree;
```

For static variables defined in File [image_reduction_composite.h](include/image_reduction_composite.h), they are all safe to be static for visualization use cases:
```c++
namespace Legion::Visualization::ImageReductionComposite {

/* These are function definitions.. */
static CompositeFunction compositePixelsNever;
static CompositeFunction compositePixelsLess;
static CompositeFunction compositePixelsEqual;
static CompositeFunction compositePixelsLEqual;
static CompositeFunction compositePixelsGreater;
static CompositeFunction compositePixelsNotEqual;
static CompositeFunction compositePixelsGEqual;
static CompositeFunction compositePixelsAlways;
static CompositeFunction blendPixelsSlowly;
static CompositeFunction blendPixelsFast__ONE__ADD__ONE_MINUS_SRC_ALPHA;

/* These variables can remain constant if only alpha blending is used. 
   For (sort-last) visualization, only alpha blending is used. */
// Because these are static we will only be able to support one blend
//  operation per Legion runtime instance at a time
// TODO remove this limitation by dynamically registering these in an array
static GLenum mGlBlendFunctionSource;
static GLenum mGlBlendFunctionDestination;
static GLenum mBlendEquation;

}
```

There are some other static variables defined in [legion_visualization.h](include/legion_visualization.h), they are safe.
```c++
// legion_visualization.h
namespace Legion::Visualization {
/* They are constants, so safe */
static const int image_region_dimensions = 3;//(width x height) x layerID
static const int max_pFields = 32; // max fields in the simulation region
}
```
