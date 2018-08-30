
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG=1                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
USE_CUDA	?= 0
USE_HDF		?= 0
USE_GASNET	?= 0
ALT_MAPPERS	?= 0		  # Compile the alternative mappers

# Put the binary file name here
OUTFILE		:= visualization_test_3
# List all the application source files here
GEN_SRC		:= ../visualization_reductions.cc visualization_test_3.cc        # .cc files
#GEN_GPU_SRC	:=		    # .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	?= -I../../include -I${LG_RT_DIR} -I${LG_RT_DIR}/legion -I${LG_RT_DIR}/realm -I..
CC_FLAGS	:=
NVCC_FLAGS	:=
GASNET_FLAGS	:=
LD_FLAGS	:= -L../../src -lImageCompositor

###########################################################################
#
#   Don't change anything below here
#
###########################################################################

include $(LG_RT_DIR)/runtime.mk

