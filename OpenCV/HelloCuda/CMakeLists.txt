cmake_minimum_required(VERSION 2.8.8)
set(CMAKE_BUILD_TYPE Debug)

project(OpenCVCudaMiniDemos C CXX)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()



# -+- OpenCV

set(OpenCV_DIR "/home/blackzafiro/Descargas/Desarrollo/OpenCV/opencv-3.2.0/build/share/OpenCV")
find_package(OpenCV 3.2 REQUIRED COMPONENTS highgui cudacodec cudaimgproc cudaarithm)
include_directories("include"
                    ${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_INSTALL_PATH}/lib)



# -+- Demos

add_executable(HelloGPU   HelloGPU.cpp)
target_link_libraries(HelloGPU ${OpenCV_LIBS})

add_executable(VideoReader   VideoReader.cpp)
target_link_libraries(VideoReader ${OpenCV_LIBS})
