cmake_minimum_required(VERSION 2.8.8)

# -+- Project

project(gpu-track C CXX)

# -*- Dependencies
set(OpenCV_DIR "/home/blackzafiro/Descargas/Desarrollo/OpenCV/opencv-3.2.0/build/share/OpenCV")

find_package(OpenCV 3.2 REQUIRED COMPONENTS highgui cudacodec cudaimgproc cudaarithm)

# -*- Indicate where include files are
include_directories(${OpenCV_INCLUDE_DIRS})

# -*- Indicate where include libraries are
link_directories(${OpenCV_INSTALL_PATH}/lib)

# -+- app

set(srcs src/gpu_track.cpp)

add_executable(gpu_track ${srcs})

# Link executable to libraries
target_link_libraries(gpu_track ${OpenCV_LIBS})
