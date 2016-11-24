
# You must configure ALTERAOCLSDKROOT to point the root directory of the Altera SDK for OpenCL
# software installation.

TARGET = MDHost
TARGET_GPU = MDHost_GPU
TARGET_CPU = MDHost_CPU


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

HEADERS = ./include
# arm cross compiler
CROSS-COMPILE = arm-linux-gnueabihf-

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG=$(shell aocl compile-config --arm) -I../common/inc
AOCL_LINK_CONFIG=$(shell aocl link-config --arm)


#GPU COMPILATION
GPU_INCLUDE = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include"
GPU_LIB = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\lib\Win32"

#OPENMP COMPILATION
SRCS_CPU = md_cpu.cpp
SRCS_CPU_FILES = $(foreach F, $(SRCS_CPU), openmp_implementation/$(F))

all :
	$(CROSS-COMPILE)g++ -I $(HEADERS) -D ALTERA $(SRCS_FILES) $(COMMON_FILES) -o $(TARGET)  $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG)

gpu :
	g++ $(SRCS_FILES) -I $(HEADERS) -D NVIDIA -I $(GPU_INCLUDE) -L $(GPU_LIB) -o $(TARGET_GPU) -lOpenCL

cpu :
	g++ $(SRCS_CPU_FILES) -I $(HEADERS) -w -O3 -o $(TARGET_CPU) -fopenmp
# Standard make targets
clean :
	@rm -f *.o $(TARGET)
	@rm -f *.o $(TARGET_CPU)
	@rm -f *.o $(TARGET_GPU)

