/* Copyright 2017 Stanford University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



#ifndef image_reduction_h
#define image_reduction_h

//#define TRACE_TASKS

//tracing for debug
#if 1
#define __TRACE {std::cout<<__FILE__<<":"<<__LINE__<<" "<<__FUNCTION__<<std::endl;}
#else
#define __TRACE
#endif

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <assert.h>

#include "legion/legion_c_util.h"
#include "KDTree.hpp"


namespace Legion {
  namespace Visualization {

    class ImageReduction {

    private:

      typedef struct {
        ImageDescriptor imageDescriptor;
        bool isAssociative;
        int compositeTaskID;
        GLenum depthFunction;
        GLenum blendFunctionSource;
        GLenum blendFunctionDestination;
        LogicalRegion image;
      } ScreenSpaceArguments;

      typedef struct {
        ImageDescriptor imageDescriptor;
        GLenum depthFunction;
        GLenum blendFunctionSource;
        GLenum blendFunctionDestination;
        GLenum blendEquation;
        GLfloat cameraDirection[image_region_dimensions];
      } CompositeArguments;

      typedef struct {
        ImageDescriptor imageDescriptor;
        int t;
      } DisplayArguments;

    public:

      enum FieldIDs {
        FID_FIELD_R = 0,
        FID_FIELD_G,
        FID_FIELD_B,
        FID_FIELD_A,
        FID_FIELD_Z,
        FID_FIELD_USERDATA,
        FID_FIELD_COLOR, // used in coloring logical region
        FID_FIELD_EXTENT // used in extent logical region
      };

      typedef float PixelField;
      static const int numPixelFields = 6;//rgbazu
      typedef float SimulationBoundsCoordinate;
      typedef size_t Stride[ImageReduction::numPixelFields][image_region_dimensions];

      /**
       * Initialize the image reduction framework.
       * Be sure to call this before starting the Legion runtime.
       *
       * @param mapperID dynamically generated mapper ID (see runtime->generate_dynamic_mapper_id)
       */
      static void preinitializeBeforeRuntimeStarts();

      /**
       * Preregister an array of simulation bounds in 3D.
       * This is optional, required if you plan to use noncommutative reductions.
       * Be sure to call this *before* starting the Legion runtime.
       *
       * @param bounds array of 6xGLfloatxnumNodes
       * @param numBounds number of simulation elements
       */
      static void preregisterSimulationBounds(SimulationBoundsCoordinate *bounds, int numBounds);

      ImageReduction(){}
      /**
       * Construct an image reduction framework based on ImageDescriptor.
       *
       * @param imageDescriptor defines dimensions of current image
       * @param ctx Legion context
       * @param runtime  Legion runtime
       */
      ImageReduction(ImageDescriptor imageDescriptor, Context ctx, HighLevelRuntime *runtime);
      /**
       * Construct an image reduction framework based on an existing partition.
       *
       * @param partition defines an existing partition with one node per subregion
       * @param imageDescriptor defines dimensions of current image
       * @param ctx Legion context
       * @param runtime  Legion runtime
       */
       ImageReduction(LogicalRegion region,
       LogicalPartition partition,
       legion_field_id_t pFields[],
       int numPFields,
       ImageDescriptor imageDescriptor,
       Context context,
       HighLevelRuntime *runtime);

      /**
       * Destroy an instance of an image reduction framework.
       */
      virtual ~ImageReduction();

      /**
       * Perform a tree reduction.
       * Be sure to call either set_blend_func or set_depth_func first.
       */
      FutureMap reduceImages(Context context, float cameraDirection[image_region_dimensions] = nullptr);
      /**
       * Move reduced image result to a display.
       *
       * @param t integer timestep
       */
      Future display(int t, Context context);

      /**
       * Provide the camera view matrix, typically from gluLookAt
       *
       * @param view a 4x4 homogeneous view matrix, typically from gluLookAt
       */
      void set_view_matrix(GLfloat view[]) {
        memcpy(mGlViewTransform, view, sizeof(mGlViewTransform));
      }

      /**
       * Define blend source and destination functions to use in subsequent reductions
       * (see glBlendFunc).
       *
       * @param sfactor source blend factor
       * @param dfactor destination blend factor
       */
      void set_blend_func(GLenum sfactor, GLenum dfactor) {
        mGlBlendFunctionSource = sfactor;
        mGlBlendFunctionDestination = dfactor;
      }

      /**
       * Specify the constant color to use with certain blend functions (see glBlendFunc)
       * @param red constant color
       * @param green constant color
       * @param blue constant color
       * @param alpha constant alpha
       */
      void set_blend_color(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) {
        mGlConstantColor[FID_FIELD_R] = red;
        mGlConstantColor[FID_FIELD_G] = green;
        mGlConstantColor[FID_FIELD_B] = blue;
        mGlConstantColor[FID_FIELD_A] = alpha;
      }

      /**
       * Specify the use with blending (this is not common, see glBlendEquation)
       * @param mode must be one of GL_FUNC_ADD, GL_FUNC_SUBTRACT, GL_FUNC_REVERSE_SUBTRACT, GL_MIN, GL_MAX
       */
      void set_blend_equation(GLenum mode) {
        assert(mode == GL_FUNC_ADD || mode == GL_FUNC_SUBTRACT || mode == GL_FUNC_REVERSE_SUBTRACT || mode == GL_MIN || mode == GL_MAX);
        mGlBlendEquation = mode;
      }

      /**
       * Define a depth operator to use in subsequent reductions.
       * For definition of depth factors (see glDepthFunc).
       * @param func depth comparison factor
       */
      void set_depth_func(GLenum func){ mDepthFunction = func; }

      /**
       * obtain raw pointers to image data
       *
       * @param imageDescriptor see legion_visualization.h
       * @param region physical region of image fragment
       * @param r return raw pointer to pixel fields
       * @param g return raw pointer to pixel fields
       * @param b return raw pointer to pixel fields
       * @param a return raw pointer to pixel fields
       * @param z return raw pointer to pixel fields
       * @param userdata return raw pointer to pixel fields
       * @param stride returns stride between successive pixels
       * @param readWrite true if read/write access
       */
      static void create_image_field_pointers(ImageDescriptor imageDescriptor,
                                              PhysicalRegion region,
                                              PixelField *&r,
                                              PixelField *&g,
                                              PixelField *&b,
                                              PixelField *&a,
                                              PixelField *&z,
                                              PixelField *&userdata,
                                              Stride stride,
                                              Runtime *runtime,
                                              Context context,
                                              bool readWrite);

      /**
       * Utility function to provide descriptive output for messages.
       *
       * @param task Legion task pointer
       */
      static std::string describe_task(const Task *task) {
        char hostname[128] = { 0 };
        if(hostname[0] == '\0') gethostname(hostname, sizeof(hostname));
        std::ostringstream output;
        output << hostname << " ";
        output << task->get_task_name() << " "
        << task->task_id << " "
        << task->get_unique_id()
        << " pid " << getpid()
        << " " << hostname
        << " (" << task->index_point.point_data[0]
        << ", " << task->index_point.point_data[1]
        << ", " << task->index_point.point_data[2]
        << ")"
        ;
        return output.str();
      }
      /**
       * obtain the compositeImage domain, useful for index launches
       */
      Domain compositeImageDomain() const {
        return mCompositeImageDomain;
      }
      /*
       * obtain the compositeImage partition
       */
      LogicalPartition compositeImagePartition() const {
        return mCompositeImagePartition;
      }
      /*
       * obtain the compositeImage color space
       */
      Domain compositeImageColorSpace() const {
        return mCompositeImageColorSpace;
      }
      /**
       * obtain the renderImage domain, useful for index launches
       */
      Domain renderImageDomain() const {
        return mRenderImageDomain;
      }
      /**
       * obtain the renderImage partition
       */
      LogicalPartition renderImagePartition() const {
        return mRenderImagePartition;
      }
      /*
       * obtain the renderImage color space
       */
      Domain renderImageColorSpace() const {
        return mRenderImageColorSpace;
      }
      /**
       * obtain the source image index space
       **/
      IndexSpace sourceIndexSpace() const {
        return mSourceIndexSpace;
      }
      /**
       * obtain the source image logical region
       */
      LogicalRegion sourceImage() const {
        return mSourceImage;
      }
      /**
       * obtain the source image fields
       **/
      void sourceImageFields(Context context, legion_field_id_t imageFields[]) const {
        std::vector<FieldID> fields;
        mRuntime->get_field_space_fields(context, mSourceImageFields, fields);
        for(unsigned i = 0; i < fields.size(); ++i) {
          imageFields[i] = fields[i];
        }
      }
      /**
       * obtain the image descriptor, pass this to the mapper
       */
      ImageDescriptor imageDescriptor() const {
        return mImageDescriptor;
      }


      static void display_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);

      static int numTreeLevels(ImageDescriptor imageDescriptor);
      static int numTreeLevels(int numImageLayers);

      static void initial_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);

      static KDNode<image_region_dimensions, long long int>* findFragmentInKDTree(PhysicalRegion fragment);

      static bool flipRegions(PhysicalRegion fragment0, PhysicalRegion fragment1, float cameraDirection[image_region_dimensions]);

      static void composite_task(const Task *task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime *runtime);



    protected:

      class CompositeProjectionFunctor : public ProjectionFunctor {
      public:
        CompositeProjectionFunctor(int offset, int multiplier, int numBounds, int id) {
          mOffset = offset;
          mMultiplier = multiplier;
          mNumBounds = numBounds;
          mID = id;
        }

        virtual LogicalRegion project(const Mappable *mappable, unsigned index,
                                      LogicalPartition upperBound,
                                      const DomainPoint &point) {
          int launchDomainLayer = point[2];
          DomainPoint remappedPoint = point;
          int remappedLayer = launchDomainLayer * mMultiplier + mOffset;
          // handle non-power of 2 simulation size
          if(mNumBounds == 0 || remappedLayer < mNumBounds) {
            remappedPoint[2] = remappedLayer;
          }

          LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(upperBound, remappedPoint);
          return result;
        }

        virtual LogicalRegion project(LogicalPartition upper_bound, const DomainPoint &point, const Domain &launch_domain) {
          assert(false);
          LogicalRegion result;
          return result;
        }

        virtual LogicalRegion project(const Mappable *mappable, unsigned index, LogicalRegion upper_bound, const DomainPoint &point) {
          assert(false);
          LogicalRegion result;
          return result;
        }

        virtual LogicalRegion project(LogicalRegion upper_bound, const DomainPoint &point, const Domain &launch_domain) {
          assert(false);
          LogicalRegion result;
          return result;
        }


        int id() const{ return mID; }
        std::string to_string() const {
          char buffer[256];
          sprintf(buffer, "CompositeProjectionFunctor id %d offset %d multiplier %d numNodes %d", mID, mOffset, mMultiplier, mNumBounds);
          return std::string(buffer);
        }

        virtual bool is_exclusive(void) const{ return true; }
        virtual unsigned get_depth(void) const{ return 0; }
        virtual bool is_functional(void) const { return true; }

      private:
        int mOffset;
        int mMultiplier;
        int mNumBounds;
        int mID;
      };


      static void createProjectionFunctors(Runtime* runtime, int numImageLayers);


      void initializeNodes(HighLevelRuntime* runtime, Context context);
      void initializeViewMatrix();
      void createTreeDomains(int numTreeLevels, Runtime* runtime, ImageDescriptor mImageDescriptor);
      FieldSpace imageFields(Context context);
      void createImageRegion(IndexSpace& indexSpace, LogicalRegion &region, Domain &domain, FieldSpace& fields, legion_field_id_t fieldID[], Context context);
      void partitionImageByDepth(LogicalRegion image, Domain &domain, LogicalPartition &partition, Context context);
      void partitionImageByImageDescriptor(LogicalRegion image, Context ctx, HighLevelRuntime* runtime, ImageDescriptor imageDescriptor);
      void partitionImageByKDTree(LogicalRegion image, LogicalPartition sourcePartition, Context ctx, HighLevelRuntime* runtime, ImageDescriptor imageDescriptor);

      void addCompositeArgumentsToArgmap(CompositeArguments *&argsPtr, int taskZ, ArgumentMap &argMap, int layer0, int layer1);

      void addRegionRequirementToCompositeLauncher(IndexTaskLauncher &launcher, int projectionFunctorID, PrivilegeMode privilege, CoherenceProperty coherence);

      static void buildKDTrees(ImageDescriptor imageDescriptor, Context ctx, HighLevelRuntime *runtime);

      static void registerTasks();

      static void addImageFieldsToRequirement(RegionRequirement &req);


      static void createImageRegionFieldPointer(LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic, PixelField> &acc,
                                          int fieldID,
                                          PixelField *&field,
                                          Rect<image_region_dimensions> imageBounds,
                                          PhysicalRegion region,
                                          ByteOffset offset[image_region_dimensions]);

      static int subtreeHeight(ImageDescriptor imageDescriptor);

      static FutureMap launchTreeReduction(ImageDescriptor imageDescriptor, int treeLevel,
                                           GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation,
                                           int compositeTaskID, LogicalPartition sourceFragmentPartition, LogicalRegion image,
                                           Runtime* runtime, Context context,
                                           int maxTreeLevel,
                                           float cameraDirection[image_region_dimensions]);


      ImageDescriptor mImageDescriptor;
      Runtime *mRuntime;
      IndexSpace mSourceIndexSpace;
      LogicalRegion mSourceImage;
      FieldSpace mSourceImageFields;
      Domain mSourceImageDomain;
      Domain mCompositeImageDomain;
      Domain mCompositeImageColorSpace;
      LogicalPartition mCompositeImagePartition;
      Domain mRenderImageDomain;
      Domain mRenderImageColorSpace;
      LogicalPartition mRenderImagePartition;
      Domain mDisplayDomain;
      Domain mSourceFragmentDomain;
      GLenum mDepthFunction;
      int mAccessorFunctorID;
      MapperID mMapperID;

    public:
      static int mNodeID;
      static std::vector<CompositeProjectionFunctor*> *mCompositeProjectionFunctor;
      static std::vector<Domain> *mHierarchicalTreeDomain;
      static const int numMatrixElements4x4 = 16;
      static GLfloat mGlViewTransform[numMatrixElements4x4];
      static PixelField mGlConstantColor[numPixelFields];
      static GLenum mGlBlendEquation;
      static GLenum mGlBlendFunctionSource;
      static GLenum mGlBlendFunctionDestination;
      static TaskID mInitialTaskID;
      static TaskID mCompositeTaskID;
      static TaskID mDisplayTaskID;
      static KDTree<image_region_dimensions, long long int>* mSimulationKDTree;
      static KDTree<image_region_dimensions, long long int>* mImageKDTree;
    };

  }
}

#endif /* image_reduction_h */
