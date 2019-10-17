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

  void createGraphicsContext(OSMesaContext &mesaCtx,
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

    /* Allocate the depth buffer. */
    depthBuffer = new GLfloat[width * height];
    if (!depthBuffer) {
      printf("Alloc depth buffer failed!\n");
      return;
    }
  }


  static void initializeRender(Camera* camera, int width, int height) {
    GLfloat afPropertiesAmbient [] = {1.00, 1.00, 1.00, 0.5};
    GLfloat afPropertiesDiffuse [] = {1.00, 1.00, 1.00, 0.5};
    GLfloat afPropertiesSpecular[] = {1.00, 1.00, 1.00, 0.5};
    glClearColor( 0.3, 0.3, 0.3, 0 );
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);
    glEnable(GL_LIGHTING);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glLightfv(GL_LIGHT0, GL_AMBIENT,  afPropertiesAmbient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  afPropertiesDiffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, afPropertiesSpecular);
    GLfloat lightPosition[] = { 1, 4, 1, 1 };
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0);
    glEnable(GL_LIGHT0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    GLfloat fovy = 20;
    GLfloat aspect = (GLfloat)width / (GLfloat)height;
    GLfloat near = 0.0;
    GLfloat far = 50.0;
    gluPerspective(fovy, aspect, near, far);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    gluLookAt(camera->from[0], camera->from[1], camera->from[2],
              camera->at[0], camera->at[1], camera->at[2],
              camera->up[0], camera->up[1], camera->up[2]);
  }



  static void cube(float edgeHalfSize) {
    glBegin(GL_QUADS);
    glVertex3f( edgeHalfSize, edgeHalfSize,-edgeHalfSize);    // Top Right Of The Quad (Top)
    glVertex3f(-edgeHalfSize, edgeHalfSize,-edgeHalfSize);    // Top Left Of The Quad (Top)
    glVertex3f(-edgeHalfSize, edgeHalfSize, edgeHalfSize);    // Bottom Left Of The Quad (Top)
    glVertex3f( edgeHalfSize, edgeHalfSize, edgeHalfSize);    // Bottom Right Of The Quad (Top)
    glVertex3f( edgeHalfSize,-edgeHalfSize, edgeHalfSize);    // Top Right Of The Quad (Bottom)
    glVertex3f(-edgeHalfSize,-edgeHalfSize, edgeHalfSize);    // Top Left Of The Quad (Bottom)
    glVertex3f(-edgeHalfSize,-edgeHalfSize,-edgeHalfSize);    // Bottom Left Of The Quad (Bottom)
    glVertex3f( edgeHalfSize,-edgeHalfSize,-edgeHalfSize);    // Bottom Right Of The Quad (Bottom)
    glVertex3f( edgeHalfSize, edgeHalfSize, edgeHalfSize);    // Top Right Of The Quad (Front)
    glVertex3f(-edgeHalfSize, edgeHalfSize, edgeHalfSize);    // Top Left Of The Quad (Front)
    glVertex3f(-edgeHalfSize,-edgeHalfSize, edgeHalfSize);    // Bottom Left Of The Quad (Front)
    glVertex3f( edgeHalfSize,-edgeHalfSize, edgeHalfSize);    // Bottom Right Of The Quad (Front)
    glVertex3f( edgeHalfSize,-edgeHalfSize,-edgeHalfSize);    // Top Right Of The Quad (Back)
    glVertex3f(-edgeHalfSize,-edgeHalfSize,-edgeHalfSize);    // Top Left Of The Quad (Back)
    glVertex3f(-edgeHalfSize, edgeHalfSize,-edgeHalfSize);    // Bottom Left Of The Quad (Back)
    glVertex3f( edgeHalfSize, edgeHalfSize,-edgeHalfSize);    // Bottom Right Of The Quad (Back)
    glVertex3f(-edgeHalfSize, edgeHalfSize, edgeHalfSize);    // Top Right Of The Quad (Left)
    glVertex3f(-edgeHalfSize, edgeHalfSize,-edgeHalfSize);    // Top Left Of The Quad (Left)
    glVertex3f(-edgeHalfSize,-edgeHalfSize,-edgeHalfSize);    // Bottom Left Of The Quad (Left)
    glVertex3f(-edgeHalfSize,-edgeHalfSize, edgeHalfSize);    // Bottom Right Of The Quad (Left)
    glVertex3f( edgeHalfSize, edgeHalfSize,-edgeHalfSize);    // Top Right Of The Quad (Right)
    glVertex3f( edgeHalfSize, edgeHalfSize, edgeHalfSize);    // Top Left Of The Quad (Right)
    glVertex3f( edgeHalfSize,-edgeHalfSize, edgeHalfSize);    // Bottom Left Of The Quad (Right)
    glVertex3f( edgeHalfSize,-edgeHalfSize,-edgeHalfSize);    // Bottom Right Of The Quad (Right)
    glEnd();
  }

  static GLfloat const alpha = 0.5;
  static GLfloat colorTable[8][4] = {
    { 1.0, 0.0, 0.0, alpha }, // red
    { 0.0, 1.0, 0.0, alpha }, // green
    { 0.0, 0.0, 1.0, alpha }, // blue
    { 1.0, 0.0, 1.0, alpha }, // purple
    { 1.0, 1.0, 0.0, alpha }, // yellow
    { 0.0, 1.0, 1.0, alpha }, // turqoise
    { 0.6, 1.0, 0.6, alpha }, // light green
    { 1.0, 1.0, 1.0, alpha } // white
  };


  static void setColor(Legion::Point<3> lo) {
    int index = lo.z + 2 * lo.y + 4 * lo.x;
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, colorTable[index]);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, colorTable[index]);
  #if 1
  {
    char buffer[256];
    sprintf(buffer, "color index %d lo %lld %lld %lld\n", index, lo.x, lo.y, lo.z);
    std::cout << buffer;
    if(index != 1) {
      GLfloat invisible[] = { 0, 0, 0, 0 };
      glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, invisible);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, invisible);
    }
  }
  #endif
  }



  void renderCube(Legion::Rect<3> bounds, ImageDescriptor* imageDescriptor, Camera* camera, unsigned char*& rgbaBuffer, float*& depthBuffer) {
    initializeRender(camera, imageDescriptor->width, imageDescriptor->height);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    float center[3] = {
      (float)(bounds.lo.x + 0.5),
      (float)(bounds.lo.y + 0.5),
      (float)(bounds.lo.z + 0.5)
    };
    glTranslatef(center[0], center[1], center[2]);
    setColor(bounds.lo);
    cube(0.333);
    glPopMatrix();
    glFinish();
  }

#ifdef __cplusplus
}
#endif
