cmake_minimum_required(VERSION 2.8.8)

# -+- Project

project(gpu-track C CXX)

# -*- Dependencies

# Custum directories
set(OpenCV_DIR "/home/blackzafiro/Descargas/Desarrollo/OpenCV/opencv-3.2.0/build/share/OpenCV")

set(srcs src/gpu_track.cpp)

#
# -*- OpenCV
#
find_package(OpenCV 3.2 REQUIRED COMPONENTS highgui cudacodec cudaimgproc cudaarithm)

# -+- Indicate where include files are
include_directories(${OpenCV_INCLUDE_DIRS}, "include")

# -+- Indicate where include libraries are
link_directories(${OpenCV_INSTALL_PATH}/lib)

#
# -*- Cuda
#
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS -O3
     -gencode arch=compute_52,code=sm_52	#GeForce GTX 960
     -gencode arch=compute_50,code=sm_50	#GeForce GTX 960M
     #-gencode arch=compute_20,code=sm_20	#GeForce GT 750
     #--relocatable-device-code true
     -rdc=true
     -G -g									#debug
)
add_definitions(-D_FORCE_INLINES)
set( CUDA_SEPARABLE_COMPILATION ON)

set(cudasrcs cusrc/linear_snake.cu)

CUDA_COMPILE(CUDA_OBJECTS
	cusrc/linear_snake.cu
	OPTIONS ${CUDA_FLAGS}
)
CUDA_ADD_LIBRARY(linear_snake
	${cudasrcs}
)

# -+- app


set(cuda_sources cusrc/linear_snake.cu)


add_executable(gpu_track ${srcs} )

# Link executable to libraries
message(${CUDA_LIBRARIES})

target_link_libraries(gpu_track ${OpenCV_LIBS} ${CUDA_LIBRARIES} linear_snake cudadevrt)
