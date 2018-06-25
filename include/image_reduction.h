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

//tracing -- remove this
//#define _T {std::cout<<__FILE__<<":"<<__LINE__<<" "<<__FUNCTION__<<std::endl;}

#include "legion_visualization.h"

#include "usec_timer.h"
#include "accessor.h"

#ifdef __APPLE__
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <iostream>
#include <sstream>


//#define TRACE_TASKS


namespace Legion {
  namespace Visualization {
    
    
    class ImageReduction {
      
    private:
      
      typedef struct {
        ImageSize imageSize;
        bool isAssociative;
        int compositeTaskID;
        GLenum depthFunction;
        GLenum blendFunctionSource;
        GLenum blendFunctionDestination;
        LogicalRegion image;
      } ScreenSpaceArguments;
      
      typedef struct {
        ImageSize imageSize;
        GLenum depthFunction;
        GLenum blendFunctionSource;
        GLenum blendFunctionDestination;
        GLenum blendEquation;
      } CompositeArguments;
      
      typedef struct {
        ImageSize imageSize;
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
      };
      
      typedef float PixelField;
      static const int numPixelFields = 6;//rgbazu
      static const int num_fragments_per_composite = 2;
      typedef float SimulationBoundsCoordinate;
      typedef ByteOffset Stride[ImageReduction::numPixelFields][image_region_dimensions];
      
      /**
       * Initialize the image reduction framework.
       * Be sure to call this before starting the Legion runtime.
       */
      static void initialize();

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
       * Construct an image reduction framework.
       *
       * @param imageSize defines dimensions of current image
       * @param ctx Legion context
       * @param runtime  Legion runtime
       */
      ImageReduction(ImageSize imageSize, Context ctx, HighLevelRuntime *runtime);
      /**
       * Destroy an instance of an image reduction framework.
       */
      virtual ~ImageReduction();
      
      /**
       * Launch a set of tasks that each receive one layer in Z of the image space.
       * Use this for example to render to the individual layers.
       *
       * @param taskID ID of task that has previously been registered with the Legion runtime
       */
      FutureMap launch_index_task_by_depth(unsigned taskID, HighLevelRuntime* runtime, Context context, void *args = NULL, int argLen = 0, bool blocking = false);
      /**
       * Launch a set of tasks for each layer in Z of the image space.
       * This does not have any region requirements, use this for initialization.
       *
       * @param taskID ID of task that has previously been registered with the Legion runtime
       */
      FutureMap launch_epoch_task_by_depth(unsigned taskID, HighLevelRuntime* runtime, Context context, void *args = NULL, int argLen = 0, bool blocking = false);
      /**
       * Perform a tree reduction using an associative commutative operator.
       * Be sure to call either set_blend_func or set_depth_func first.
       */
      FutureMap reduce_associative_commutative();
      /**
       * Perform a tree reduction using an associative noncommutative operator.
       * Be sure to call either set_blend_func or set_depth_func first.
       * Be sure to call preregisterSimulationBounds before starting the Legion runtime.
       */
      FutureMap reduce_associative_noncommutative();
      /**
       * Perform a pipeline reduction using a nonassociative commutative operator.
       * Be sure to call either set_blend_func or set_depth_func first.
       */
      FutureMap reduce_nonassociative_commutative();
      /**
       * Perform a pipeline reduction using a nonassociative noncommutative operator.
       * Be sure to call either set_blend_func or set_depth_func first.
       * Be sure to call preregisterSimulationBounds before starting the Legion runtime.
       */
      FutureMap reduce_nonassociative_noncommutative();
      /**
       * Move reduced image result to a display.
       *
       * @param t integer timestep
       */
      Future display(int t);
      
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
       * @param imageSize see legion_visualization.h
       * @param region physical region of image fragment
       * @param r return raw pointer to pixel fields
       * @param g return raw pointer to pixel fields
       * @param b return raw pointer to pixel fields
       * @param a return raw pointer to pixel fields
       * @param z return raw pointer to pixel fields
       * @param userdata return raw pointer to pixel fields
       * @param stride returns stride between successive pixels
       */
      static void create_image_field_pointers(ImageSize imageSize,
                                              PhysicalRegion region,
                                              PixelField *&r,
                                              PixelField *&g,
                                              PixelField *&b,
                                              PixelField *&a,
                                              PixelField *&z,
                                              PixelField *&userdata,
                                              Stride stride,
                                              Runtime *runtime,
                                              Context context);
      
      /**
       * Utility function to provide descriptive output for messages.
       *
       * @param task Legion task pointer
       */
      static std::string describe_task(const Task *task) {
        std::ostringstream output;
        output << task->get_task_name() << " " << task->get_unique_id()
        << " (" << task->index_point.point_data[0]
        << ", " << task->index_point.point_data[1]
        << ", " << task->index_point.point_data[2]
        << ")"
        ;
        return output.str();
      }
      
      static void display_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
      
      static int numTreeLevels(ImageSize imageSize);
      
      static void initial_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
      
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
          
#if 0
          {std::cout<< to_string() << " for task " << mappable->as_task()->get_unique_id()
            << " remaps launch point "<<point<<" to "<<remappedPoint<<std::endl;}
#endif
          
          LogicalRegion result = Legion::Runtime::get_runtime()->get_logical_subregion_by_color(upperBound, remappedPoint);
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
        
      private:
        int mOffset;
        int mMultiplier;
        int mNumBounds;
        int mID;
      };
      
      static CompositeProjectionFunctor* getCompositeProjectionFunctor(int nodeID, int maxTreeLevel, int level);

      static CompositeProjectionFunctor* makeCompositeProjectionFunctor(int offset, int numBounds, int nodeID, int level, int numLevels, Runtime* runtime);
      
      
      static void storeMySimulationBounds(SimulationBoundsCoordinate* values, int nodeID, int numNodes);
      
      static SimulationBoundsCoordinate* retrieveMySimulationBounds(int nodeID);
      
      static void storeMyNodeID(int nodeID, int numNodes);
      
      static void createProjectionFunctors(int nodeID, int numBounds, Runtime* runtime, ImageSize imageSize);
      
      
      
      void initializeNodes(HighLevelRuntime* runtime, Context context);
      void initializeViewMatrix();
      void createTreeDomains(int nodeID, int numTreeLevels, Runtime* runtime, ImageSize mImageSize);
      FieldSpace imageFields();
      void createImage(LogicalRegion &region, Domain &domain);
      void partitionImageByDepth(LogicalRegion image, Domain &domain, LogicalPartition &partition);
      void partitionImageByFragment(LogicalRegion image, Domain &domain, LogicalPartition &partition);
      
      FutureMap reduceAssociative();
      FutureMap reduceNonassociative();
      
      void addCompositeArgumentsToArgmap(CompositeArguments *&argsPtr, int taskZ, ArgumentMap &argMap, int layer0, int layer1);
      
      void addRegionRequirementToCompositeLauncher(IndexTaskLauncher &launcher, int projectionFunctorID, PrivilegeMode privilege, CoherenceProperty coherence);

      static void registerTasks();
      
      static void addImageFieldsToRequirement(RegionRequirement &req);
      
      
      static void createImageFieldPointer(LegionRuntime::Accessor::RegionAccessor<LegionRuntime::Accessor::AccessorType::Generic, PixelField> &acc,
                                          int fieldID,
                                          PixelField *&field,
                                          Rect<image_region_dimensions> imageBounds,
                                          PhysicalRegion region,
                                          ByteOffset offset[image_region_dimensions]);
      
      static int subtreeHeight(ImageSize imageSize);
      
      static FutureMap launchTreeReduction(ImageSize imageSize, int treeLevel,
                                           GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation,
                                           int compositeTaskID, LogicalPartition sourceFragmentPartition, LogicalRegion image,
                                           Runtime* runtime, Context context,
                                           int nodeID, int maxTreeLevel);
      
      static FutureMap launchPipelineReduction();
      
      static int subdomainToCompositeIndex(SimulationBoundsCoordinate *bounds, int scale);
      
      
      ImageSize mImageSize;
      Context mContext;
      Runtime *mRuntime;
      LogicalRegion mSourceImage;
      Domain mSourceImageDomain;
      Domain mDepthDomain;
      Domain mCompositePipelineDomain;
      Domain mDisplayDomain;
      Domain mSourceFragmentDomain;
      LogicalPartition mDepthPartition;
      LogicalPartition mSourceFragmentPartition;
      GLenum mDepthFunction;
      int mAccessorFunctorID;
      int mLocalCopyOfNodeID;
      
    public:
      static const int fieldsPerSimulationBounds = 2 * image_region_dimensions;
      static int* mNodeID;
      static SimulationBoundsCoordinate *mSimulationBounds;
      static int mNumSimulationBounds;
      static SimulationBoundsCoordinate mXMax, mXMin, mYMax, mYMin, mZMax, mZMin;
      static int mNodeCount;
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
    };
    
  }
}

#endif /* image_reduction_h */
