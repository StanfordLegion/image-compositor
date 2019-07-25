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

#include "visualization_reductions.h"

namespace Legion {
  namespace Visualization {
    
    const GLenum depthFuncs[] = {
      GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS
    };
    
    const GLenum blendFuncs[] = {
      GL_ZERO, GL_ONE, GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_DST_COLOR, GL_ONE_MINUS_DST_COLOR, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR, GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA,
    };
    
    const GLenum blendEquations[] = {
      GL_FUNC_ADD, GL_FUNC_SUBTRACT, GL_FUNC_REVERSE_SUBTRACT, GL_MIN, GL_MAX
    };
    
    const int numDepthFuncs = sizeof(depthFuncs) / sizeof(depthFuncs[0]);
    const int numBlendFuncs = sizeof(blendFuncs) / sizeof(blendFuncs[0]);
    const int numBlendEquations = sizeof(blendEquations) / sizeof(blendEquations[0]);
    
    static std::string depthFuncToString(GLenum depthFunc) {
      switch(depthFunc) {
        case GL_NEVER: return "GL_NEVER";
        case GL_LESS: return "GL_LESS";
        case GL_EQUAL: return "GL_EQUAL";
        case GL_LEQUAL: return "GL_LEQUAL";
        case GL_GREATER: return "GL_GREATER";
        case GL_NOTEQUAL: return "GL_NOTEQUAL";
        case GL_GEQUAL: return "GL_GEQUAL";
        case GL_ALWAYS: return "GL_ALWAYS";
      }
      return "none";
    }
    
    static std::string blendFuncToString(GLenum blendFunc) {
      switch(blendFunc) {
        case GL_ZERO: return "GL_ZERO";
        case GL_ONE: return "GL_ONE";
        case GL_SRC_COLOR: return "GL_SRC_COLOR";
        case GL_ONE_MINUS_SRC_COLOR: return "GL_ONE_MINUS_SRC_COLOR";
        case GL_DST_COLOR: return "GL_DST_COLOR";
        case GL_ONE_MINUS_DST_COLOR: return "GL_ONE_MINUS_DST_COLOR";
        case GL_SRC_ALPHA: return "GL_SRC_ALPHA";
        case GL_ONE_MINUS_SRC_ALPHA: return "GL_ONE_MINUS_SRC_ALPHA";
        case GL_DST_ALPHA: return "GL_DST_ALPHA";
        case GL_ONE_MINUS_DST_ALPHA: return "GL_ONE_MINUS_DST_ALPHA";
        case GL_CONSTANT_COLOR: return "GL_CONSTANT_COLOR";
        case GL_ONE_MINUS_CONSTANT_COLOR: return "GL_ONE_MINUS_CONSTANT_COLOR";
        case GL_CONSTANT_ALPHA: return "GL_CONSTANT_ALPHA";
        case GL_ONE_MINUS_CONSTANT_ALPHA: return "GL_ONE_MINUS_CONSTANT_ALPHA";
      }
      return "none";
    }
    
    
    static std::string blendEquationToString(GLenum mode) {
      switch(mode) {
        case GL_FUNC_ADD: return "GL_FUNC_ADD";
        case GL_FUNC_SUBTRACT: return "GL_FUNC_SUBTRACT";
        case GL_FUNC_REVERSE_SUBTRACT: return "GL_FUNC_REVERSE_SUBTRACT";
        case GL_MIN: return "GL_MIN";
        case GL_MAX: return "GL_MAX";
      }
      return "none";
    }
    
    
#ifdef DEBUG
    static void dumpImage(ImageReduction::PixelField *rr: case ImageReduction::PixelField*gg, ImageReduction::PixelField*bb, ImageReduction::PixelField*aa, ImageReduction::PixelField*zz, ImageReduction::PixelField*uu, ImageReduction::Stride stride, char *text) {
      std::cout << std::endl;
      std::cout << text << std::endl;
      for(int i = 0; i < 10; ++i) {
        std::cout << text << " pixel " << i << ": ";
        std::cout << rr[0] << "\t" << gg[0] << "\t" << bb[0] << "\t" << aa[0] << "\t" << zz[0] << "\t" << uu[0] << std::endl;
        ImageReductionComposite::increment(rr, gg, bb, aa, zz, uu, stride);
      }
    }
#endif
    
    
    typedef ImageReduction::PixelField* Image;
    
    
    static char* paintFileName(char *buffer, int taskID) {
      sprintf(buffer, "/tmp/paint.%d", taskID);
      return buffer;
    }
    
    static Image loadReferenceImageFromFile(int taskID, ImageDescriptor imageDescriptor) {
      char fileName[256];
      FILE *inputFile = fopen(paintFileName(fileName, taskID), "rb");
      Image result = new ImageReduction::PixelField[imageDescriptor.pixelsPerLayer() * ImageReduction::numPixelFields];
      fread(result, sizeof(ImageReduction::PixelField), imageDescriptor.pixelsPerLayer() * ImageReduction::numPixelFields, inputFile);
      fclose(inputFile);
      return result;
    }
    
    static void saveImage(int taskID, ImageDescriptor imageDescriptor, Image image) {
      char fileName[256];
      FILE *outputFile = fopen(paintFileName(fileName, taskID), "wb");
      fwrite(image, sizeof(ImageReduction::PixelField), imageDescriptor.pixelsPerLayer() * ImageReduction::numPixelFields, outputFile);
      fclose(outputFile);
    }
    
    
    // generate Image contents
    //
    
    // this is used to generate a contiguous buffer of test data
    static void paintImage(ImageDescriptor imageDescriptor, int taskID, Image &image) {
      image = new ImageReduction::PixelField[imageDescriptor.pixelsPerLayer() * ImageReduction::numPixelFields];
      Image imagePtr = image;
      ImageReduction::PixelField zValue = taskID % imageDescriptor.numImageLayers;
      
      for(int row = 0; row < imageDescriptor.height; ++row) {
        for(int column = 0; column < imageDescriptor.width; ++column) {
          *imagePtr++ = row;
          *imagePtr++ = column;
          *imagePtr++ = taskID;
          *imagePtr++ = taskID;
          *imagePtr++ = zValue;
          *imagePtr++ = taskID;
          zValue = (zValue + 1);
          zValue = (zValue >= imageDescriptor.numImageLayers) ? 0 : zValue;
        }
      }
    }
    
    
    // this is used to generate the test contents for the logical region
    static void paintRegion(ImageDescriptor imageDescriptor,
                            ImageReduction::PixelField *r,
                            ImageReduction::PixelField *g,
                            ImageReduction::PixelField *b,
                            ImageReduction::PixelField *a,
                            ImageReduction::PixelField *z,
                            ImageReduction::PixelField *userdata,
                            Legion::Visualization::ImageReduction::Stride stride,
                            int taskID) {
      
      Image image;
      paintImage(imageDescriptor, taskID, image);
      Image imagePtr = image;
      
      for(int row = 0; row < imageDescriptor.height; ++row) {
        for(int column = 0; column < imageDescriptor.width; ++column) {
          *r = *imagePtr++;
          *g = *imagePtr++;
          *b = *imagePtr++;
          *a = *imagePtr++;
          *z = *imagePtr++;
          *userdata = *imagePtr++;
          
          ImageReductionComposite::increment(r, g, b, a, z, userdata, stride);
        }
      }
      
      delete [] image;
    }
    
    
    
    void generate_image_data_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime) {
      
      UsecTimer render(ImageReduction::describe_task(task) + ":");
      render.start();
      PhysicalRegion image = regions[0];
      ImageDescriptor imageDescriptor = ((ImageDescriptor *)task->args)[0];
      
      ImageReduction::PixelField *r, *g, *b, *a, *z, *userdata;
      ImageReduction::Stride stride;
      ImageReduction::create_image_field_pointers(imageDescriptor, image, r, g, b, a, z, userdata, stride, runtime, ctx, true);
      
      Domain indexSpaceDomain = runtime->get_index_space_domain(ctx, image.get_logical_region().get_index_space());
      LegionRuntime::Arrays::Rect<image_region_dimensions> imageBounds = indexSpaceDomain.get_rect<image_region_dimensions>();
      
      int taskID = imageBounds.lo[2];
      paintRegion(imageDescriptor, r, g, b, a, z, userdata, stride, taskID);
      render.stop();
      cout << render.to_string() << endl;
    }
    
    
    static bool verifyImageReduction(int pixelID, char *fieldName, ImageReduction::PixelField expected, ImageReduction::PixelField actual) {
      if(expected != actual) {
        std::cerr << "verification failure at pixel " << pixelID << " field " << fieldName << " expected " << expected << " saw " << actual << std::endl;
        return false;
      }
      return true;
    }
    
    static int verifyImage(ImageDescriptor imageDescriptor, Image expected, ImageReduction::PixelField *r, ImageReduction::PixelField *g, ImageReduction::PixelField *b, ImageReduction::PixelField *a, ImageReduction::PixelField *z, ImageReduction::PixelField *userdata, ImageReduction::Stride stride) {
      // expected comes from a file and has contiguous data
      // the other pointers are from a logical region and are separated by stride[i][0]
      
      const int maxFailuresBeforeAbort = 10;
      int failures = 0;
      for(int i = 0; i < imageDescriptor.pixelsPerLayer(); ++i) {
        if(!verifyImageReduction(i, (char*)"r", *expected++, *r)) {
          failures++;
        }
        if(!verifyImageReduction(i, (char*)"g", *expected++, *g)) {
          failures++;
        }
        if(!verifyImageReduction(i, (char*)"b", *expected++, *b)) {
          failures++;
        }
        if(!verifyImageReduction(i, (char*)"a", *expected++, *a)) {
          failures++;
        }
        if(!verifyImageReduction(i, (char*)"z", *expected++, *z)) {
          failures++;
        }
        if(!verifyImageReduction(i, (char*)"userdata", *expected++, *userdata)) {
          failures++;
        }
        ImageReductionComposite::increment(r, g, b, a, z, userdata, stride);
        if(failures >= maxFailuresBeforeAbort) {
          std::cerr << "too many failures, aborting verification" << std::endl;
          break;
        }
      }
      return failures;
    }
    
    
    
    int verify_composited_image_data_task(const Task *task,
                                          const std::vector<PhysicalRegion> &regions,
                                          Context ctx, Runtime *runtime) {
      coord_t layer = task->index_point[2];
      if(layer == 0) {
        PhysicalRegion image = regions[0];
        ImageDescriptor imageDescriptor = ((ImageDescriptor *)task->args)[0];
        Image expected = (Image)((char*)task->args + sizeof(imageDescriptor));
        ImageReduction::PixelField *r, *g, *b, *a, *z, *userdata;
        ImageReduction::Stride stride;
        ImageReduction::create_image_field_pointers(imageDescriptor, image, r, g, b, a, z, userdata, stride, runtime, ctx, false);
        return verifyImage(imageDescriptor, expected, r, g, b, a, z, userdata, stride);
      }
      return 0;
    }
    
    static int verifyAccumulatorMatchesResult(ImageReduction &imageReduction, Image expected, ImageDescriptor imageDescriptor, HighLevelRuntime* runtime, Context context) {
      int totalSize = imageDescriptor.pixelsPerLayer() * ImageReduction::numPixelFields * sizeof(ImageReduction::PixelField);
      FutureMap futures = imageReduction.launch_task_everywhere(VERIFY_COMPOSITED_IMAGE_DATA_TASK_ID, runtime, context, expected, totalSize, true);
      DomainPoint origin = Point<image_region_dimensions>::ZEROES();
      int failures = futures[origin].get<int>();
      return failures;
    }
    
    
    
    
    
    static void compositeTwoImages(Image image0, Image image1, ImageDescriptor imageDescriptor,
                                   GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation) {
      
      // Reuse the composite functions from the ImageReductionComposite class.
      // This means those functions will not be independently tested.
      // Those functions are simple enough to be verified by inspection.
      // There seems no point in writing a duplicate set of functions for testing.
      
      ImageReductionComposite::CompositeFunction *compositeFunction =
      ImageReductionComposite::compositeFunctionPointer(depthFunc, blendFuncSource, blendFuncDestination, blendEquation);
      
      // these images have contiguous data and do not come from Legion regions, they come from files
      ImageReduction::PixelField *r0In = image0;
      ImageReduction::PixelField *g0In = r0In + 1;
      ImageReduction::PixelField *b0In = g0In + 1;
      ImageReduction::PixelField *a0In = b0In + 1;
      ImageReduction::PixelField *z0In = a0In + 1;
      ImageReduction::PixelField *userdata0In = z0In + 1;
      
      ImageReduction::PixelField *r1In = image1;
      ImageReduction::PixelField *g1In = r1In + 1;
      ImageReduction::PixelField *b1In = g1In + 1;
      ImageReduction::PixelField *a1In = b1In + 1;
      ImageReduction::PixelField *z1In = a1In + 1;
      ImageReduction::PixelField *userdata1In = z1In + 1;
            
      ImageReduction::Stride stride;
      for(int i = 0; i < ImageReduction::numPixelFields; ++i) {
        stride[i][0] = 1 * ImageReduction::numPixelFields;
      }
      
      compositeFunction(r0In, g0In, b0In, a0In, z0In, userdata0In, r1In, g1In, b1In, a1In, z1In, userdata1In,
                        r0In, g0In, b0In, a0In, z0In, userdata0In, imageDescriptor.pixelsPerLayer(), stride, stride);
    }
    
    
    static std::string testDescription(std::string testLabel, ImageReduction &imageReduction, ImageDescriptor imageDescriptor,
                                       GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination,
                                       GLenum blendEquation) {
      return testLabel + " " + imageDescriptor.toString() + "\n"
      + "depth " + depthFuncToString(depthFunc)
      + " blendSource " + blendFuncToString(blendFuncSource)
      + " blendDestination " + blendFuncToString(blendFuncDestination)
      + " blendEquation " + blendEquationToString(blendEquation);
    }
    
    
    static void verifyTestResult(std::string testLabel, ImageReduction &imageReduction, ImageDescriptor imageDescriptor,
                                 GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation, Image expected,
                                 HighLevelRuntime* runtime, Context context) {
      
      int numFailures = verifyAccumulatorMatchesResult(imageReduction, expected, imageDescriptor, runtime, context);
      std::string description = testDescription(testLabel, imageReduction, imageDescriptor, depthFunc, blendFuncSource,
                                                blendFuncDestination, blendEquation);
      if(numFailures == 0) {
        std::cout << "SUCCESS: " << description << std::endl;
      } else {
        std::cerr << "FAILURES: " << numFailures << " " << description << std::endl;
      }
    }
    
    
    static Image treeReduction(int treeLevel, int maxTreeLevel, int leftLayer,
                               ImageReduction &imageReduction, ImageDescriptor imageDescriptor,
                               GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation) {
      
      Image image0 = NULL;
      Image image1 = NULL;
      int layer0 = leftLayer;
      int layerOffset = (int)powf(2.0f, maxTreeLevel - treeLevel - 1);
      int layer1 = layer0 + layerOffset;
      
      if(treeLevel == maxTreeLevel - 1) {
        image0 = loadReferenceImageFromFile(layer0, imageDescriptor);
        image1 = loadReferenceImageFromFile(layer1, imageDescriptor);
      } else {
        image0 = treeReduction(treeLevel + 1, maxTreeLevel, layer0, imageReduction, imageDescriptor,
                               depthFunc, blendFuncSource, blendFuncDestination, blendEquation);
        image1 = treeReduction(treeLevel + 1, maxTreeLevel, layer1, imageReduction, imageDescriptor,
                               depthFunc, blendFuncSource, blendFuncDestination, blendEquation);
      }

      compositeTwoImages(image0, image1, imageDescriptor, depthFunc, blendFuncSource, blendFuncDestination, blendEquation);
      
      delete [] image1;
      return image0;
    }
    
    
    static void savePaintedImages(ImageDescriptor imageDescriptor) {
      for(int taskID = 0; taskID < imageDescriptor.numImageLayers; ++taskID) {
        
        Image image;
        paintImage(imageDescriptor, taskID, image);
        saveImage(taskID, imageDescriptor, image);
        delete [] image;
      }
    }
    
    static void verifyAssociativeTestResult(std::string testLabel, ImageReduction &imageReduction, ImageDescriptor imageDescriptor,
                                            GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation,
                                            HighLevelRuntime* runtime, Context context) {
      savePaintedImages(imageDescriptor);
      int maxTreeLevel = ImageReduction::numTreeLevels(imageDescriptor);
      Image expected = treeReduction(0, maxTreeLevel, 0, imageReduction, imageDescriptor,
                                     depthFunc, blendFuncSource, blendFuncDestination, blendEquation);
      verifyTestResult(testLabel, imageReduction, imageDescriptor,
                       depthFunc, blendFuncSource, blendFuncDestination, blendEquation, expected, runtime, context);
      delete [] expected;
    }
    
    
    static Image pipelineReduction(ImageReduction &imageReduction, ImageDescriptor imageDescriptor,
                                   GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation) {
      Image expected = loadReferenceImageFromFile(0, imageDescriptor);
      for(int layer = 1; layer < imageDescriptor.numImageLayers; ++layer) {
        Image nextImage = loadReferenceImageFromFile(layer, imageDescriptor);
        compositeTwoImages(expected, nextImage, imageDescriptor, depthFunc, blendFuncSource, blendFuncDestination, blendEquation);
      }
      return expected;
    }
    
    static void verifyNonassociativeTestResult(std::string testLabel, ImageReduction &imageReduction, ImageDescriptor imageDescriptor,
                                               GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation,
                                               HighLevelRuntime* runtime, Context context) {
      savePaintedImages(imageDescriptor);
      Image expected = pipelineReduction(imageReduction, imageDescriptor,
                                         depthFunc, blendFuncSource, blendFuncDestination, blendEquation);
      verifyTestResult(testLabel, imageReduction, imageDescriptor,
                       depthFunc, blendFuncSource, blendFuncDestination, blendEquation, expected, runtime, context);
      delete [] expected;
    }
    
    
    static void paintImages(ImageDescriptor imageDescriptor, Context context, Runtime *runtime, ImageReduction &imageReduction) {
      void* args = &imageDescriptor;
      int argLen = sizeof(ImageDescriptor);
      FutureMap futures = imageReduction.launch_task_everywhere(GENERATE_IMAGE_DATA_TASK_ID, runtime, context, args, argLen, /*blocking*/true);
    }
    
    
    static void prepareTest(ImageReduction &imageReduction, ImageDescriptor imageDescriptor, Context context, Runtime *runtime, GLenum depthFunc,
                            GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation, std::string testLabel) {
      std::string marker = "=====";
      std::cout << marker << std::endl;
      std::cout << marker << std::endl;
      std::cout << marker << std::endl;
      std::cout << testDescription(testLabel, imageReduction, imageDescriptor, depthFunc, blendFuncSource, blendFuncDestination, blendEquation) << std::endl;
      
      paintImages(imageDescriptor, context, runtime, imageReduction);
      imageReduction.set_depth_func(depthFunc);
      imageReduction.set_blend_func(blendFuncSource, blendFuncDestination);
      imageReduction.set_blend_equation(blendEquation);
    }
    
    
    
    
    void testAssociative(ImageReduction &imageReduction,
                         ImageDescriptor imageDescriptor, Context context, Runtime *runtime,
                         GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation) {
      std::string testLabel = "associative,commutative";
      
      std::cout << testDescription(testLabel, imageReduction, imageDescriptor, depthFunc, blendFuncSource, blendFuncDestination, blendEquation) << std::endl;
      
      prepareTest(imageReduction, imageDescriptor, context, runtime, depthFunc, blendFuncSource, blendFuncDestination, blendEquation, testLabel);
      UsecTimer reduceCommutative("image reduction " + testLabel);
      reduceCommutative.start();
      FutureMap futureMap = imageReduction.reduce_associative_commutative(context);
      futureMap.wait_all_results();
      reduceCommutative.stop();
      std::cout << reduceCommutative.to_string() << std::endl;
      verifyAssociativeTestResult(testLabel, imageReduction, imageDescriptor, depthFunc, blendFuncSource, blendFuncDestination, blendEquation, runtime, context);
      
      return;//stop here
      
      testLabel = "associative,noncommutative";
      futureMap = imageReduction.reduce_associative_noncommutative(context);
      futureMap.wait_all_results();
      verifyAssociativeTestResult(testLabel, imageReduction, imageDescriptor, depthFunc, blendFuncSource, blendFuncDestination, blendEquation, runtime, context);
    }
    
    
    
    void testNonassociative(ImageReduction &imageReduction,
                            ImageDescriptor imageDescriptor, Context context, Runtime *runtime,
                            GLenum depthFunc, GLenum blendFuncSource, GLenum blendFuncDestination, GLenum blendEquation) {
      
      return;//not working
      
      std::string testLabel = "nonassociative,commutative";
      prepareTest(imageReduction, imageDescriptor, context, runtime, depthFunc, blendFuncSource, blendFuncDestination, blendEquation, testLabel);
      FutureMap futureMap;
      futureMap = imageReduction.reduce_nonassociative_commutative(context);
      futureMap.wait_all_results();
      verifyNonassociativeTestResult(testLabel, imageReduction, imageDescriptor, depthFunc, blendFuncSource, blendFuncDestination, blendEquation, runtime, context);
      
      testLabel = "nonassociative,noncommutative";
      futureMap = imageReduction.reduce_nonassociative_noncommutative(context);
      futureMap.wait_all_results();
      verifyNonassociativeTestResult(testLabel, imageReduction, imageDescriptor, depthFunc, blendFuncSource, blendFuncDestination, blendEquation, runtime, context);
    }
    
    const int numDomainNodesX = 2;
    const int numDomainNodesY = 2;
    const int numDomainNodesZ = 1;
    const int numDomainNodes = numDomainNodesX * numDomainNodesY * numDomainNodesZ;
    
    void preregisterSimulationBounds(int numSimulationBoundsX, int numSimulationBoundsY, int numSimulationBoundsZ) {
      
      // call this before starting the Legion runtime
      
      const int fieldsPerBound = 6;
      float *bounds = new float[numDomainNodes * fieldsPerBound];
      float *boundsPtr = bounds;
      
      // construct a simple regular simulation domain
      for(int x = 0; x < numSimulationBoundsX; ++x) {
        for(int y = 0; y < numSimulationBoundsY; ++y) {
          for(int z = 0; z < numSimulationBoundsZ; ++z) {
            *boundsPtr++ = x;
            *boundsPtr++ = y;
            *boundsPtr++ = z;
            *boundsPtr++ = x + 1;
            *boundsPtr++ = y + 1;
            *boundsPtr++ = z + 1;
          }
        }
      }
      
      Legion::Visualization::ImageReduction::preregisterSimulationBounds(bounds, numDomainNodes);
      delete [] bounds;
    }
    
    void top_level_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
    
  }
}


