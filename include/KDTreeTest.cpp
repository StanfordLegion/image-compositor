//
// KDTreeTest.cpp
//
// Balanced KDTree implementation tester for image compositor framework.
//

#include "KDTree.hpp"

int main(int argc, char* argv[]) {
  typedef int ElementType[3];
  
  const unsigned numPoints = 2;
  unsigned numElements = 0;
  ElementType elements[numPoints * numPoints * numPoints];
  
  for(unsigned z = 0; z < numPoints; ++z) {
    for(unsigned y = 0; y < numPoints; ++y) {
      for(unsigned x = 0; x < numPoints; ++x) {
        ElementType element;
        element[0] = x;
        element[1] = y;
        element[2] = z;
        memcpy(elements + numElements++, element, sizeof(ElementType));
      }
    }
  }
  
  KDTree<3, int, ElementType> kdTree(elements, numElements);
  kdTree.dump();
}


