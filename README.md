# image-compositor

## Introduction
This is an image compositor framework for Legion.  It allows Legion applications to render data in situ and in parallel and reduce the resulting plurality of images down to a single image for display.

(./doc/isav_workshop_sc17_presentation.m4v "Video Illustration")

<video controls="controls">
<source type="video/m4v" src="doc/isav_workshop_sc17_presentation.m4v"></source>
</video>

## Requirements
The framework supports the standard OpenGL Depth and Blend compositing functions in a distributed environment.
These are sufficient for most scientific visualization purposes such as surface and volume rendering.
The framework supports compositing functions of the following algebraic types:
associative-commutative, associative-noncommutative, nonassociative-noncommutative.
Surface rendering uses the associative-commutative depth functions, while volume rendering uses the associative-noncommutative blend functions.

Note: following the philosophy of "don't build it until there is a customer" the framework has only implemented the associate-commutative reductions.
See "Blending todo" for notes on how to implement the other reductions,

## Code organization

----
├── examples
│   └── visualization
├── include
├── src
└── test
           ├── visualization_test_1
           ├── visualization_test_2
           └── visualization_test_3
----

### framework
#### mapper
### tests
#### running tests
### examples
#### running examples

## Use cases
### Tests and examples
### C++ Legion application
### Regent application with rendering in C++/OpenGL
#### Soleil-x
### Regent application with rendering in Regent

## Blending todo

