## -*- Makefile -*-

# Variables básicas
CXX = g++
CPPFLAGS = -ggdb -fpic -v
LDFLAGS =

GPUXX = nvcc
GPUXXFLAGS = -c -v

# Fuentes
SOURCE = GpuDraw.cpp
SOURCE_CU = gpuDraw.cu
SOURCES = $(SOURCE_CU) $(SOURCE)
#OBJECTS = $(SOURCE:.cpp=.o) $(SOURCE_CU:.cu=.o)
OBJECTS = $(SOURCE_CU:.cu=.o)
#OBJECTS = $(SOURCE:.cpp=.o)

# Ejecutable
EXEC = GpuDraw
OBJ_CU = gpuDraw

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
	@echo -e '\e[0;32mGenerating executable...\e[0m'
#	$(CXX) $(CPPFLAGS) $(SOURCE) -o $@ $(LDFLAGS)
	$(CXX) $(CPPFLAGS) $(OBJECTS) $(SOURCE) -o $@ $(LDFLAGS)

#.cpp.o:
#	@echo -e '\e[0;32mCompiling objects...\e[0m'
#	$(CXX) $(CPPFLAGS) $< -o $@
#	@echo -e '\n'

%.o: %.cu
	@echo -e '\e[0;32mCompiling cuda kernels...\e[0m'
	$(GPUXX) $(GPUXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo -e '\n'

clean:
	rm $(OBJECTS) $(EXEC)

print:
	@echo $(SOURCES)
	@echo $(OBJECTS)