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
  ElementType colorMap[numElements];
  kdTree.colorMap(colorMap);
  for(unsigned i = 0; i < numElements; ++i) {
    std::cout << colorMap[i][0] << " " << colorMap[i][1] << " " << colorMap[i][2] << std::endl;
  }
  
  for(unsigned i = 0; i < numElements; ++i) {
    KDNode<3, int, ElementType>* node = kdTree.find(elements[i]);
    if(node == nullptr) {
      std::cout << "could not find " << elements[i][0] << " " << elements[i][1] << " " << elements[i][2] << std::endl;
    } else {
      std::cout << elements[i][0] << " " << elements[i][1] << " " << elements[i][2] << " == ";
      std::cout << node->mValue[0] << " " << node->mValue[1] << " " << node->mValue[2] << std::endl;
    }
  }
}


