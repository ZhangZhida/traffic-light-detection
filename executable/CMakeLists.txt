cmake_minimum_required(VERSION 2.8.12)
project(executable)

add_executable(traffic_light_detect traffic_light_detect.cpp)
target_link_libraries(traffic_light_detect traffic_light_utils ${OpenCV_LIBS})

add_executable(get_point_color get_point_color.cpp)
target_link_libraries(get_point_color traffic_light_utils ${OpenCV_LIBS})