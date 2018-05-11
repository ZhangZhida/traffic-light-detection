cmake_minimum_required(VERSION 2.8.12)
project(executable)

add_executable(traffic_light_detect traffic_light_detect.cpp)
target_link_libraries(traffic_light_detect ${OpenCV_LIBS})