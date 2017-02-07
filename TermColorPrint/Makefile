## -*- Makefile -*-

# Variables básicas
CXX = g++
CPPFLAGS = -ggdb -fpic
LDFLAGS =

# Ejecutable
EXEC = demo

# Fuentes
INCLUDE_DIR = include
INCLUDES = include/TermColorPrint.h
SOURCE = src/$(EXEC).cpp
SOURCES = src/TermColorPrint.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Flags
CPPFLAGS += -I$(INCLUDE_DIR)
LDFLAGS +=

# Compile

all: $(EXEC)

$(EXEC): $(OBJECTS) $(SOURCE)
	$(CXX) $(CPPFLAGS) $(OBJECTS) $(SOURCE) -o $@ $(LDFLAGS)

.cpp.o:
	$(CXX) -c $(CPPFLAGS) $< -o $@

clean:
	rm $(OBJECTS) $(EXEC)

print:
	@echo $(SOURCES)
	@echo $(OBJECTS)

# Dependencies
src/TermColorPrint.o: include/TermColorPrint.h