//
// renderCube.cc
//

// This program requires the OpenGL and GLUT libraries
// You can obtain them for free from http://www.opengl.org

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include "OpenGL/glu.h"
#else
#include "GL/osmesa.h"
#endif

#include "GL/glu.h"
#include "GL/glut.h"

#include "render.h"
#include "legion.h"
#include "legion_c.h"
#include "legion_visualization.h"
#include "image_reduction_mapper.h"


#ifdef __cplusplus
extern "C" {
#endif
  
  static void createGraphicsContext(OSMesaContext &mesaCtx,
                                    GLubyte* &rgbaBuffer,
                                    GLfloat* &depthBuffer,
                                    int width,
                                    int height) {
#if OSMESA_MAJOR_VERSION * 100 + OSMESA_MINOR_VERSION >= 305
    /* specify Z, stencil, accum sizes */
    mesaCtx = OSMesaCreateContextExt(GL_RGBA, 32, 0, 0, NULL);
#else
    mesaCtx = OSMesaCreateContext(GL_RGBA, NULL);
#endif
    if (!mesaCtx) {
      printf("OSMesaCreateContext failed!\n");
      return;
    }
    
    
    /* Allocate the image buffer */
    const int fieldsPerPixel = 4;
    rgbaBuffer = new GLubyte[width * height * fieldsPerPixel];
    if (!rgbaBuffer) {
      printf("Alloc image buffer failed!\n");
      return;
    }
    
    /* Bind the buffer to the context and make it current */
    if (!OSMesaMakeCurrent(mesaCtx, rgbaBuffer, GL_UNSIGNED_BYTE, width, height)) {
      printf("OSMesaMakeCurrent failed!\n");
      return;
    }
    
    {
      int z, s, a;
      glGetIntegerv(GL_DEPTH_BITS, &z);
      glGetIntegerv(GL_STENCIL_BITS, &s);
      glGetIntegerv(GL_ACCUM_RED_BITS, &a);
    }
    
    /* Allocate the depth buffer. */
    depthBuffer = new GLfloat[width * height];
    if (!depthBuffer) {
      printf("Alloc depth buffer failed!\n");
      return;
    }
  }
  
  static void destroyGraphicsContext(OSMesaContext mesaCtx) {
    /* destroy the context */
    OSMesaDestroyContext(mesaCtx);
  }
  
  
  static void initializeRender(Camera* camera, int width, int height) {
    glClearColor( 0, 0, 0, 1 );
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat afPropertiesAmbient [] = {1.00, 1.00, 1.00, 0.5};//alpha=0.5, translucent
    GLfloat afPropertiesDiffuse [] = {1.00, 1.00, 1.00, 0.5};
    GLfloat afPropertiesSpecular[] = {1.00, 1.00, 1.00, 0.5};
    glLightfv(GL_LIGHT0, GL_AMBIENT,  afPropertiesAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  afPropertiesDiffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, afPropertiesSpecular);
    GLfloat lightPosition[] = { 1, 4, 1, 1 };
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glViewport(0, 0, width, height);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluLookAt(camera->from[0], camera->from[1], camera->from[2],
              camera->at[0], camera->at[1], camera->at[2],
              camera->up[0], camera->up[1], camera->up[2]);
  }
  
  
  static void drawCube(Rect<3> bounds) {
    float edgeSize = (float)(bounds.hi.x - bounds.lo.x);
    float center[3] = { (float)(bounds.lo.x + bounds.hi.x) / 2,
      (float)(bounds.lo.y + bounds.hi.y) / 2,
      (float)(bounds.lo.z + bounds.hi.z) / 2 };
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(center[0], center[1], center[2]);
    glutSolidCube(edgeSize / 2);
    glPopMatrix();
  }
  
  void renderCube(Rect<3> bounds, ImageDescriptor* imageDescriptor, Camera* camera, unsigned char*& rgbaBuffer, float*& depthBuffer) {
    OSMesaContext mesaCtx;
    createGraphicsContext(mesaCtx, rgbaBuffer, depthBuffer, imageDescriptor->width, imageDescriptor->height);
    initializeRender(camera, imageDescriptor->width, imageDescriptor->height);
    drawCube(bounds);
    destroyGraphicsContext(mesaCtx);
  }
  
#ifdef __cplusplus
}
#endif
