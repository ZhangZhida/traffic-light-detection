cmake_minimum_required(VERSION 2.8.12)
project(traffic-light-detection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 11)

#include(3rdparty/dlib/cmake)
include_directories(include)
# equals to
# add_subdirectory(../dlib dlib_build)


#set(CMAKE_PREFIX_PATH "/home/cuizhou/Downloads/opencv-3.3.0/build_cuda")



set(CMAKE_PREFIX_PATH "/home/zhida/Downloads/OpenCV-3.4.1/opencv-3.4.1/build")
find_package(OpenCV 3.4.1 REQUIRED)

add_subdirectory(src)
add_subdirectory(executable)