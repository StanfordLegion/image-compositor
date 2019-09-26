//
// KDTree.hpp
//
// Balanced KDTree implementation for image compositor framework.
//

#ifndef __KDTREE_HPP__
#define __KDTREE_HPP__

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "legion_visualization.h"

using namespace Legion::Visualization;
using namespace LegionRuntime::HighLevel;

template<int N, typename SplitterType, typename ElementType>
class KDNode {
public:
  KDNode(KDNode<N, SplitterType, ElementType>* parent) {
    mIsLeaf = false;
    mLevel = 0;
    mLeft = mRight = nullptr;
    mParent = parent;
  }
  KDNode(ElementType element, unsigned level, KDNode<N, SplitterType, ElementType>* parent) {
    mIsLeaf = true;
    memcpy(mValue, element, sizeof(ElementType));
    mLevel = level;
    mLeft = mRight = nullptr;
    mParent = parent;
  }
  
  unsigned mLevel;
  SplitterType mSplitter;
  bool mIsLeaf;
  ElementType mValue;//only for leaf nodes
  KDNode<N, SplitterType, ElementType>* mParent;
  KDNode<N, SplitterType, ElementType>* mLeft;
  KDNode<N, SplitterType, ElementType>* mRight;
};


template<unsigned N, typename DataType, typename ElementType>
class KDTree {
public:
  KDTree(ElementType elements[], unsigned numElements) {
    mRoot = buildKDTree(elements, numElements, 0);
    mNumElements = numElements;
  }
  
  KDNode<N, DataType, ElementType>* root() const{ return mRoot; }
  
  void dump() const { dumpRecursive(mRoot); };
  
  void colorMap(int coloring[], ImageDescriptor imageDescriptor) {
    unsigned zero = 0;
    colorMapRecursive(mRoot, coloring, zero, imageDescriptor);
    
  }
  
  KDNode<N, DataType, ElementType>* find(ElementType element) const {
    return findRecursive(mRoot, element);
  }
  
  unsigned size() const{ return mNumElements; }
  
private:
  KDNode<N, DataType, ElementType>* mRoot;
  unsigned mNumElements;
  
  static int compare0(const void* element1, const void* element2) {
    ElementType* e1 = (ElementType*)element1;
    ElementType* e2 = (ElementType*)element2;
    if(*e1[0] < *e2[0]) return -1;
    if(*e1[0] > *e2[0]) return +1;
    return 0;
  }
  static int compare1(const void* element1, const void* element2) {
    ElementType* e1 = (ElementType*)element1;
    ElementType* e2 = (ElementType*)element2;
    if(*e1[1] < *e2[1]) return -1;
    if(*e1[1] > *e2[1]) return +1;
    return 0;
  }
  static int compare2(const void* element1, const void* element2) {
    ElementType* e1 = (ElementType*)element1;
    ElementType* e2 = (ElementType*)element2;
    if(*e1[2] < *e2[2]) return -1;
    if(*e1[2] > *e2[2]) return +1;
    return 0;
  }

  void sortElements(ElementType elements[], unsigned numElements, unsigned index) {
    switch(index) {
      case 0: qsort(elements, numElements, sizeof(ElementType), compare0);
        break;
      case 1: qsort(elements, numElements, sizeof(ElementType), compare1);
        break;
      case 2: qsort(elements, numElements, sizeof(ElementType), compare2);
        break;
    }
    
  }
  
  KDNode<N, DataType, ElementType>* buildKDTree(
                                                ElementType elements[],
                                                unsigned numElements,
                                                unsigned level,
                                                KDNode<N, DataType, ElementType>* parent = nullptr) {
    if(numElements == 1) {
      return new KDNode<N, DataType, ElementType>(elements[0], level, parent);
    }
    sortElements(elements, numElements, level % N);
    unsigned medianIndex = numElements / 2;
    KDNode<N, DataType, ElementType>* node = new KDNode<N, DataType, ElementType>(parent);
    node->mSplitter = elements[medianIndex][level % N];
    node->mLevel = level;
    node->mLeft = buildKDTree(elements, medianIndex, level + 1, node);
    node->mRight = buildKDTree(elements + medianIndex, medianIndex, level + 1, node);
    return node;
  }
  
  void dumpRecursive(KDNode<N, DataType, ElementType>* node) const {
    if(node->mIsLeaf) {
      for(unsigned i = 0; i < N; ++i) {
        std::cout << node->mValue[i] << " ";
      }
      std::cout << std::endl;
    } else {
      dumpRecursive(node->mLeft);
      dumpRecursive(node->mRight);
    }
  }
  
  void colorMapRecursive(KDNode<N, DataType,
                         ElementType>* node, int coloring[],
                         unsigned& index,
                         ImageDescriptor imageDescriptor) {
    if(node->mIsLeaf) {
      // return a permutation of the image subregions according to the KD tree order
      int imageIndex = node->mValue[2 * image_region_dimensions];
      Point<image_region_dimensions> p0(0, 0, imageIndex);
      Point<image_region_dimensions> p1(imageDescriptor.width, imageDescriptor.height, imageIndex);
      Rect<image_region_dimensions> rect(p0, p1);
      coloring[index++] = imageIndex; // Domain::from_rect<image_region_dimensions>(rect);
    } else {
      colorMapRecursive(node->mLeft, coloring, index, imageDescriptor);
      colorMapRecursive(node->mRight, coloring, index, imageDescriptor);;
    }
  }

  KDNode<N, DataType, ElementType>* findRecursive(KDNode<N, DataType, ElementType>* node,
                                                  ElementType element) const {
    if(node->mIsLeaf) {
      for(unsigned i = 0; i < N; ++i) {
        if(node->mValue[i] != element[i]) return nullptr;
      }
      return node;
    }
    KDNode<N, DataType, ElementType>* left = findRecursive(node->mLeft, element);
    if(left != nullptr) return left;
    KDNode<N, DataType, ElementType>* right = findRecursive(node->mRight, element);
    if(right != nullptr) return right;
    return nullptr;
  }
  
};

#endif // __KDTREE_HPP__
