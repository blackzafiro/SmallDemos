## -*- Makefile -*-

# Variables básicas
CXX = g++
CPPFLAGS = -ggdb -fpic -v
LDFLAGS =

# Fuentes
SOURCES = VideoReader.cpp
#OBJECTS = $(SOURCES:.cpp=.o)

# Ejecutable
EXEC = VideoReader

export PKG_CONFIG_PATH:=$(PKG_CONFIG_PATH):/home/blackzafiro/Descargas/Desarrollo/OpenCV/opencv-3.2.0/build/lib/pkgconfig

# OpenCV
CPPFLAGS += `pkg-config --cflags opencv`
LDFLAGS += `pkg-config --libs opencv`
CV_LIB = -L/home/blackzafiro/Descargas/Desarrollo/OpenCV/opencv-3.2.0/build/share/OpenCV/3rdparty/lib

#all: $(SOURCES) $(EXEC)

#$(EXEC): $(OBJECTS)
#	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
 
#.cpp.o:
#	$(CXX) $(CPPFLAGS) $< -o $@

compile: $(SOURCES)
	$(CXX) $(CPPFLAGS) VideoReader.cpp -o $(EXEC) $(LDFLAGS) $(CV_LIB)

clean:
	rm $(OBJECTS) $(EXEC)
