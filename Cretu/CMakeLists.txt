cmake_minimum_required(VERSION 2.8.8)

### Project

project(cretu_project C CXX)

### Dependencies

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)


### CUDA Kernels

set(CUDAK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
MESSAGE( ${OpenCV_INCLUDE_DIRS} )

# Chose gencode arch and code from
# https://en.wikipedia.org/wiki/CUDA#Supported_GPUs
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
	-O3
	#-gencode arch=compute_52,code=sm_52	#GeForce GTX 960
	-gencode arch=compute_50,code=sm_50		#GeForce GTX 960M
	#-gencode arch=compute_20,code=sm_20	#GeForce GT 750
)
#add_compile_options(-c -v)
add_definitions(-D_FORCE_INLINES)
cuda_include_directories(${OpenCV_INCLUDE_DIRS} ${INC_DIR})
#cuda_compile(CudaKernel ${CUDAK_DIR}/segment.cu)
cuda_add_library(CudaKernels ${CUDAK_DIR}/segment.cu)
add_definitions(-DGPU_OPENCV_ENABLE)


### APP

file(GLOB srcs ./src/*.cpp ${INC_DIR}/*.h ./cuda/*.cu)

# Executable called "cretu_app" built from sourcefiles
add_executable(cretu_app ${srcs})

# Indicate where include files are
include_directories(${INC_DIR})

# Link executable to libraries
target_link_libraries(cretu_app
						CudaKernels
						${OpenCV_LIBS})
