#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <stdlib.h>
#include <png.h>

class PNGImage
{
public:
  int width;
  int height;
  png_bytep *row_pointers;

  png_bytep px(int x, int y)
  {
    png_bytep row = row_pointers[y];
    png_bytep p = &(row[x * 4]);
    return p;
  }

  png_byte R(int x, int y)
  {
    return px(x, y)[0];
  }

  png_byte G(int x, int y)
  {
    return px(x, y)[1];
  }

  png_byte B(int x, int y)
  {
    return px(x, y)[2];
  }

  png_byte A(int x, int y)
  {
    return px(x, y)[3];
  }

  ~PNGImage()
  {
    for(int y = 0; y < height; y++)
    {
      free(row_pointers[y]);
    }
    free(row_pointers);
  }
};

void read_png_file(const char *filename, PNGImage *image);
void write_png_file(char *filename, int width, int height,
                    float *R, float *G, float *B, float *A);

#endif
