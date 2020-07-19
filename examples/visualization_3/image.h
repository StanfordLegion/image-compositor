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

  unsigned int R(int x, int y)
  {
    return (unsigned int)row_pointers[y][x];
  }

  unsigned int G(int x, int y)
  {
    return (unsigned int)row_pointers[y][x+1];
  }

  unsigned int B(int x, int y)
  {
    return (unsigned int)row_pointers[y][x+2];
  }

  unsigned int A(int x, int y)
  {
    return (unsigned int)row_pointers[y][x+3];
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

void read_png_file(char *filename, PNGImage *image);

#endif
