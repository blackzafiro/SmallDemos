## -*- Makefile -*-

# Variables básicas
CXX = g++
CPPFLAGS = -ggdb -fpic -v
LDFLAGS =

GPUXX = nvcc
GPUXXFLAGS = -c -v
CUDA_HOME = /usr/local/cuda

# Fuentes
SOURCE = ProcessVideoFrames.cpp
SOURCE_CU = imgdiff.cu
SOURCES = $(SOURCE_CU) $(SOURCE)
#OBJECTS = $(SOURCE:.cpp=.o) $(SOURCE_CU:.cu=.o)
OBJECTS = $(SOURCE_CU:.cu=.o)

# Ejecutable
EXEC = ProcessVideoFrames
OBJ_CU = Diff 

# OpenCV location
OPENCV_HOME=/home/blackzafiro/Descargas/Desarrollo/OpenCV/opencv-3.2.0/build

export PKG_CONFIG_PATH:=$(PKG_CONFIG_PATH):$(OPENCV_HOME)/lib/pkgconfig

# OpenCV
CPPFLAGS += `pkg-config --cflags opencv`
GPUXXFLAGS += `pkg-config --cflags opencv`
CV_LIB = -L$(OPENCV_HOME)/share/OpenCV/3rdparty/lib -L$(CUDA_HOME)/lib64
LDFLAGS += `pkg-config --libs opencv` $(CV_LIB) -lcudart

all: $(SOURCES) $(EXEC)

$(EXEC): $(OBJECTS) $(SOURCE)
	$(CXX) $(CPPFLAGS) $(OBJECTS) $(SOURCE) -o $@ $(LDFLAGS)

.cpp.o:
	$(CXX) $(CPPFLAGS) $< -o $@

%.o: %.cu
	$(GPUXX) $(GPUXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm $(OBJECTS) $(EXEC)

print:
	@echo $(SOURCES)
	@echo $(OBJECTS)