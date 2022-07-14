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
#include "legion.h"

using namespace Legion::Visualization;

namespace Legion{
  namespace Visualization{

typedef struct {
  Legion::Point<image_region_dimensions> color;
  Legion::Rect<image_region_dimensions> extent;
  Legion::Rect<image_region_dimensions> extent2;
  Legion::Rect<image_region_dimensions> world_space_bounds;
} KDTreeValue;


template<int N, typename SplitterType>
class KDNode {
public:
  KDNode(KDNode<N, SplitterType>* parent) {
    mIsLeaf = false;
    mLevel = 0;
    mLeft = mRight = nullptr;
    mParent = parent;
  }
  KDNode(KDTreeValue element, unsigned level, KDNode<N, SplitterType>* parent) {
    mIsLeaf = true;
    mValue = element;
    mLevel = level;
    mLeft = mRight = nullptr;
    mParent = parent;
  }

  unsigned mLevel;
  SplitterType mSplitter;
  bool mIsLeaf;
  KDTreeValue mValue;//only for leaf nodes
  KDNode<N, SplitterType>* mParent;
  KDNode<N, SplitterType>* mLeft;
  KDNode<N, SplitterType>* mRight;
};


template<unsigned N, typename DataType>
class KDTree {
public:
  KDTree(KDTreeValue elements[], unsigned numElements) {
    mRoot = buildKDTree(elements, numElements, 0);
    mNumElements = numElements;
  }

  KDNode<N, DataType>* root() const{ return mRoot; }

  void dump() const { dumpRecursive(mRoot); };

  void getColorMap(Legion::Point<image_region_dimensions>* coloring) {
    unsigned zero = 0;
    getColorMapRecursive(mRoot, coloring, zero);
  }

  void getExtent(Legion::Rect<image_region_dimensions>* extents) {
    unsigned zero = 0;
    getExtentRecursive(mRoot, extents, zero);
  }

  KDNode<N, DataType>* find(KDTreeValue element) const {
    return findRecursive(mRoot, element);
  }

  int size() const{ return mNumElements; }

private:
  KDNode<N, DataType>* mRoot;
  int mNumElements;

  static int compare0(const void* element1, const void* element2) {
    KDTreeValue* e1 = (KDTreeValue*)element1;
    KDTreeValue* e2 = (KDTreeValue*)element2;
    if(e1->extent.lo[0] < e2->extent.lo[0]) return -1;
    if(e1->extent.lo[0] > e2->extent.lo[0]) return +1;
    return 0;
  }
  static int compare1(const void* element1, const void* element2) {
    KDTreeValue* e1 = (KDTreeValue*)element1;
    KDTreeValue* e2 = (KDTreeValue*)element2;
    if(e1->extent.lo[1] < e2->extent.lo[1]) return -1;
    if(e1->extent.lo[1] > e2->extent.lo[1]) return +1;
    return 0;
  }
  static int compare2(const void* element1, const void* element2) {
    KDTreeValue* e1 = (KDTreeValue*)element1;
    KDTreeValue* e2 = (KDTreeValue*)element2;
    if(e1->extent.lo[2] < e2->extent.lo[2]) return -1;
    if(e1->extent.lo[2] > e2->extent.lo[2]) return +1;
    return 0;
  }

  void sortElements(KDTreeValue elements[], unsigned numElements, unsigned index) {
    switch(index) {
      case 0: qsort(elements, numElements, sizeof(KDTreeValue), compare0);
        break;
      case 1: qsort(elements, numElements, sizeof(KDTreeValue), compare1);
        break;
      case 2: qsort(elements, numElements, sizeof(KDTreeValue), compare2);
        break;
    }

  }

  KDNode<N, DataType>* buildKDTree(KDTreeValue elements[],
                                   unsigned numElements,
                                   unsigned level,
                                   KDNode<N, DataType>* parent = nullptr) {
    if(numElements == 1) {
      return new KDNode<N, DataType>(elements[0], level, parent);
    }
    sortElements(elements, numElements, level % N);
    unsigned medianIndex = numElements / 2;
    KDNode<N, DataType>* node = new KDNode<N, DataType>(parent);
    node->mSplitter = elements[medianIndex].extent.lo[level % N];
    node->mLevel = level;
    node->mLeft = buildKDTree(elements, medianIndex, level + 1, node);
    node->mRight = buildKDTree(elements + medianIndex, numElements - medianIndex, level + 1, node);
    return node;
  }

  void dumpRecursive(KDNode<N, DataType>* node) const {
    if(node->mIsLeaf) {
      std::cout << node->mValue.extent << " " << node->mValue.color << " " << node->mValue.extent2 << std::endl;
    } else {
      dumpRecursive(node->mLeft);
      dumpRecursive(node->mRight);
    }
  }

  void getColorMapRecursive(KDNode<N, DataType>* node,
                            Legion::Point<image_region_dimensions>* coloring,
                            unsigned& index) {
    if(node->mIsLeaf) {
      coloring[index++] = node->mValue.color;
    } else {
      getColorMapRecursive(node->mLeft, coloring, index);
      getColorMapRecursive(node->mRight, coloring, index);
    }
  }

  void getExtentRecursive(KDNode<N, DataType>* node,
                            Legion::Rect<image_region_dimensions>* extents,
                            unsigned& index) {
    if(node->mIsLeaf) {
      extents[index++] = node->mValue.extent;
    } else {
      getExtentRecursive(node->mLeft, extents, index);
      getExtentRecursive(node->mRight, extents, index);
    }
  }

  KDNode<N, DataType>* findRecursive(KDNode<N, DataType>* node,
                                     KDTreeValue element) const {
    if(node->mIsLeaf) {
      if(node->mValue.extent == element.extent) return node;
      return nullptr;
    }
    KDNode<N, DataType>* left = findRecursive(node->mLeft, element);
    if(left != nullptr) return left;
    KDNode<N, DataType>* right = findRecursive(node->mRight, element);
    if(right != nullptr) return right;
    return nullptr;
  }

};

}
}


#endif // __KDTREE_HPP__
