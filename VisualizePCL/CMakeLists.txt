CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.1)

### Project
PROJECT(VisualizePCL)

### Dependencies
set(PCL_DIR "/home/blackzafiro/Descargas/Desarrollo/pcl-trunk/build/share/pcl-1.8")
list(APPEND CMAKE_MODULE_PATH "${PCL_DIR}")

find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization)

### APP

file(GLOB srcs ./src/*.cpp)

# Executable
add_executable(VisualizeCube ${srcs})

# Indicate where include files are
include_directories(${PCL_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Link executable to libraries
target_link_libraries(VisualizeCube
                      ${PCL_COMMON_LIBRARIES}
                      ${PCL_IO_LIBRARIES}
                      ${PCL_VISUALIZATION_LIBRARIES})
                      