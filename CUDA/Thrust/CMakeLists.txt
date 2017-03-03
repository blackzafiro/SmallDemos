cmake_minimum_required(VERSION 2.8.8)

# -+- Project

project(thrust-sample C CXX)

# -*- Dependencies

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS -O3
     -gencode arch=compute_52,code=sm_52	#GeForce GTX 960
     -gencode arch=compute_50,code=sm_50	#GeForce GTX 960M
     -gencode arch=compute_20,code=sm_20	#GeForce GT 750
)
add_definitions(-D_FORCE_INLINES)

# -+- Cuda app

set(cuda_sources src/hello_thrust.cu)

cuda_add_executable(thrust-sample ${cuda_sources})

