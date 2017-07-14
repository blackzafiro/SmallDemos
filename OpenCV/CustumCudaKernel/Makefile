## -*- Makefile -*-

# Variables básicas
CXX = nvcc
CPPFLAGS = -v
LDFLAGS =

# Fuentes
SOURCE = main.cu

# Ejecutable
EXEC = MakeGray

# OpenCV location must be stored in environment variable OPENCV_HOME
export PKG_CONFIG_PATH:=$(PKG_CONFIG_PATH):$(OPENCV_HOME)/lib/pkgconfig

# OpenCV
CPPFLAGS += `pkg-config --cflags opencv`
LDFLAGS += `pkg-config --libs opencv`
CV_LIB = -L$(OPENCV_HOME)/share/OpenCV/3rdparty/lib

all: compile

compile: $(SOURCE)
	$(CXX) $(CPPFLAGS) $(SOURCE) -o $(EXEC) $(LDFLAGS) $(CV_LIB)

clean:
	rm $(OBJECTS) $(EXEC)
