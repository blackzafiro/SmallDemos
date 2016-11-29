## -*- Makefile -*-

# Variables básicas
CXX = g++
CPPFLAGS = -ggdb -fpic -v
LDFLAGS =

#GPUXX = nvcc
#GPUXXFLAGS = -c -v

# Fuentes
SOURCE = GpuDraw.cpp
#SOURCE_CU = imgdiff.cu
#SOURCES = $(SOURCE_CU) $(SOURCE)
SOURCES = $(SOURCE)
#OBJECTS = $(SOURCE:.cpp=.o) $(SOURCE_CU:.cu=.o)
#OBJECTS = $(SOURCE_CU:.cu=.o)
OBJECTS = $(SOURCE:.cpp=.o)

# Ejecutable
EXEC = GpuDraw
#OBJ_CU = Diff 

# OpenCV location
OPENCV_HOME=/home/blackzafiro/Descargas/Programacion/opencv/opencv-3.1.0/build

export PKG_CONFIG_PATH:=$(PKG_CONFIG_PATH):$(OPENCV_HOME)/lib/pkgconfig

# OpenCV
CPPFLAGS += `pkg-config --cflags opencv`
GPUXXFLAGS += `pkg-config --cflags opencv`
CV_LIB = -L$(OPENCV_HOME)/share/OpenCV/3rdparty/lib
LDFLAGS += `pkg-config --libs opencv` $(CV_LIB) -lcudart

all: $(SOURCES) $(EXEC)

$(EXEC): $(OBJECTS) $(SOURCE)
	$(CXX) $(CPPFLAGS) $(OBJECTS) $(SOURCE) -o $@ $(LDFLAGS)

.cpp.o:
	$(CXX) $(CPPFLAGS) $< -o $@

#%.o: %.cu
#	$(GPUXX) $(GPUXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm $(OBJECTS) $(EXEC)

print:
	@echo $(SOURCES)
	@echo $(OBJECTS)