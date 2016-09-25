
# You must configure ALTERAOCLSDKROOT to point the root directory of the Altera SDK for OpenCL
# software installation.

TARGET = MCHost
TARGET_GPU = MCHost_GPU

# Where is the Altera SDK for OpenCL software?
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
endif

SRCS = main.cpp
SRCS_FILES = $(foreach F, $(SRCS), host/src/$(F))
COMMON_FILES = ../common/src/AOCL_Utils.cpp

# arm cross compiler
CROSS-COMPILE = arm-linux-gnueabihf-

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG=$(shell aocl compile-config --arm) -I../common/inc
AOCL_LINK_CONFIG=$(shell aocl link-config --arm)


#GPU COMPILATION
GPU_INCLUDE = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"
GPU_LIB = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\lib\Win32"

# Make it all!
all :
	$(CROSS-COMPILE)g++ -D ALTERA $(SRCS_FILES) $(COMMON_FILES) -o $(TARGET)  $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG)

gpu :
	g++ $(SRCS_FILES) -I $(GPU_INCLUDE) -L $(GPU_LIB) -o $(TARGET_GPU) -lOpenCL
# Standard make targets
clean :
	@rm -f *.o $(TARGET)
