#
# Makefile for visualization_2 example
#

ifndef LEGION_DIR
  $(error LEGION_DIR is not set)
endif

ifndef IMAGE_COMPOSITOR_DIR
  $(error IMAGE_COMPOSITOR_DIR is not set)
endif

CXXFLAGS += -std=c++11 -ggdb -O0 -Wall -Werror -fno-strict-aliasing -I$(LEGION_DIR)/runtime -I$(LEGION_DIR)/runtime/legion -I$(IMAGE_COMPOSITOR_DIR)/include -I$(LEGION_DIR)/runtime/realm -I$(LEGION_DIR)/bindings/regent -march=native -DDEBUG_LEGION -DGLAPIENTRY=

export INCLUDE_PATH := .;$(IMAGE_COMPOSITOR_DIR)/include;$(LEGION_DIR)/runtime/realm

REGENT := $(LEGION_DIR)/language/regent.py -g

LINK_FLAGS += -g
LINK_FLAGS += -L$(LEGION_DIR)/bindings/regent -lregent
LINK_FLAGS += -L$(IMAGE_COMPOSITOR_DIR)/src -lImageCompositor

all: visualization_2.exec

clean:
	rm -f visualization_2.exec visualization_2.so render.o visualization_2_mapper.o
	rm -f image.*.tga


visualization_2.exec: visualization_2.so render.o renderCube.o visualization_2_mapper.o
	$(CXX) -o $@ $^ $(LINK_FLAGS) -lGLU -lOSMesa

visualization_2.so: visualization_2.rg
	$(REGENT) visualization_2.rg $(REGENT_FLAGS)

render.o: render.cc render.h 
	$(CXX) $(CXXFLAGS) -c -o $@ $<

renderCube.o: renderCube.cc render.h 
	$(CXX) $(CXXFLAGS) -c -o $@ $<

visualization_2_mapper.o: visualization_2_mapper.cc visualization_2_mapper.h 
	$(CXX) $(CXXFLAGS) -c -o $@ $<
