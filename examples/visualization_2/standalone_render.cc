//
// standalone_render.cc
//
// used for standalone debug of rendering
//

#include <iostream>
#include <stdio.h>
#include <math.h>

#include "legion.h"
#include "legion_visualization.h"

#include "GL/osmesa.h"

using namespace Legion;

typedef struct { float from[3]; float at[3]; float up[3]; } Camera;



#ifdef __cplusplus
extern "C" {
#endif


void createGraphicsContext(OSMesaContext &mesaCtx,
                                  GLubyte* &rgbaBuffer,
                                  GLfloat* &depthBuffer,
                                  int width,
                                  int height);
void renderCube(Legion::Rect<3> bounds, ImageDescriptor* imageDescriptor,
  Camera* camera, unsigned char*& rgbaBuffer, float*& depthBuffer);



int gFrameNumber = 0;


int main(int argc, char *argv[]) {

  if(argc != 4) {
    std::cout << "invoke with three coordinates of bounds.lo" << std::endl;
    std::cout << argv[0] << " 1 1 0" << std::endl;
    exit(-1);
  }

  Legion::Rect<3> bounds;
  bounds.lo[0] = atoi(argv[1]);
  bounds.lo[1] = atoi(argv[2]);
  bounds.lo[2] = atoi(argv[3]);

  Visualization::ImageDescriptor imageDescriptor = { 1280, 720, 8 };
  Camera camera;
  camera.up[0] = 0;
  camera.up[1] = 1;
  camera.up[2] = 0;
  camera.from[0] = cos(0) * 6;
  camera.from[1] = 1.5;
  camera.from[2] = sin(0) * 6;
  camera.at[0] = 1;
  camera.at[1] = 1;
  camera.at[2] = 1;

  // Create MESA context
  OSMesaContext mesaCtx;
  unsigned char* rgbaBuffer;
  float* depthBuffer;
  createGraphicsContext(mesaCtx, rgbaBuffer, depthBuffer,
    imageDescriptor.width, imageDescriptor.height);

  renderCube(bounds, &imageDescriptor, &camera, rgbaBuffer, depthBuffer);

  char filename[1024];
  sprintf(filename, "image_%d_%d_%d.%05d.tga",
    (int)bounds.lo.x, (int)bounds.lo.y, (int)bounds.lo.z, gFrameNumber++);
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
  fputc (imageDescriptor.width & 0xff, f);      /* Image Width */
  fputc ((imageDescriptor.width>>8) & 0xff, f);
  fputc (imageDescriptor.height & 0xff, f);     /* Image Height   */
  fputc ((imageDescriptor.height>>8) & 0xff, f);
  fputc (0x18, f);     /* Pixel Depth, 0x18 => 24 Bits  */
  fputc (0x20, f);     /* Image Descriptor  */
  fclose(f);

  f = fopen(filename, "ab");  /* reopen in binary append mode */

  for(int y = imageDescriptor.height - 1; y >= 0; y--) {
    for(int x = 0; x < imageDescriptor.width; ++x) {
      int index = x + y * imageDescriptor.width;
      GLubyte b_ = rgbaBuffer[index * 4 + 2];
      fputc(b_, f); /* write blue */
      GLubyte g_ = rgbaBuffer[index * 4 + 1] ;
      fputc(g_, f); /* write green */
      GLubyte r_ = rgbaBuffer[index * 4];
      fputc(r_, f);   /* write red */
    }
  }
  fclose(f);
  std::cout << "wrote image " << filename << std::endl;

}

#ifdef __cplusplus
}
#endif
