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

template<int N, typename SplitterType, typename ElementType>
class KDNode {
public:
  KDNode() {
    mIsLeaf = false;
    mLevel = 0;
    mLeft = mRight = nullptr;
  }
  KDNode(ElementType element, unsigned level) {
    mIsLeaf = true;
    memcpy(mValue, element, sizeof(ElementType));
    mLevel = level;
    mLeft = mRight = nullptr;
  }
  
  unsigned mLevel;
  SplitterType mSplitter;
  bool mIsLeaf;
  ElementType mValue;//only for leaf nodes
  KDNode<N, SplitterType, ElementType>* mLeft;
  KDNode<N, SplitterType, ElementType>* mRight;
};


template<unsigned N, typename DataType, typename ElementType>
class KDTree {
public:
  KDTree(ElementType elements[], unsigned numElements) {
    mRoot = buildKDTree(elements, numElements, 0);
  }
  
  KDNode<N, DataType, ElementType>* root() const{ return mRoot; }
  
  void dump() const { dumpRecursive(mRoot); };
  
  void colorMap(ElementType colorMap[]) const { colorMapRecursive(mRoot, colorMap); }
  
private:
  KDNode<N, DataType, ElementType>* mRoot;
  
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
  
  KDNode<N, DataType, ElementType>* buildKDTree(ElementType elements[], unsigned numElements, unsigned level) {
    if(numElements == 1) return new KDNode<N, DataType, ElementType>(elements[0], level);
    sortElements(elements, numElements, level % N);
    unsigned medianIndex = numElements / 2;
    KDNode<N, DataType, ElementType>* node = new KDNode<N, DataType, ElementType>();
    node->mSplitter = elements[medianIndex][level % N];
    node->mLevel = level;
    node->mLeft = buildKDTree(elements, medianIndex, level + 1);
    node->mRight = buildKDTree(elements + medianIndex, medianIndex, level + 1);
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
  
  void colorMapRecursive(KDNode<N, DataType, ElementType>* node, ElementType*& nextElement) const {
    if(node->mIsLeaf) {
      for(unsigned i = 0; i < N; ++i) {
        (*nextElement)[i] = node->mValue[i];
      }
      nextElement++;
    } else {
      colorMapRecursive(node->mLeft, nextElement);
      colorMapRecursive(node->mRight, nextElement);
    }
  }

  
};

#endif // __KDTREE_HPP__
