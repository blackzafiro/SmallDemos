cmake_minimum_required(VERSION 2.8.8)

project(cretu_project C CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB srcs ./*.cpp ./*.h*)

add_executable(cretu_app ${srcs})
target_link_libraries(sample_app ${OpenCV_LIBS})
