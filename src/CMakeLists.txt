cmake_minimum_required(VERSION 2.8.12)
project(src)


#aux_source_directory(PvaDetector PVASRCS)
#message(${PVASRCS})
#add_library(pva SHARED ${PVASRCS})

#aux_source_directory(utils UTILSSRCS)
#message(${UTILSSRCS})
#add_library(tools SHARED ${UTILSSRCS})
#target_link_libraries(tools ${OpenCV_LIBS})# added by myf


#aux_source_directory(myfUtils MyfUtils_SRC)
#add_library( myfUtils
#        SHARED
#        ${MyfUtils_SRC} myfUtils/CropToPieces.cpp myfUtils/CropToPieces.h myfUtils/NmsThreshFilterBoxes.cpp myfUtils/NmsThreshFilterBoxes.h)
#target_link_libraries( myfUtils
#        ${OpenCV_LIBS})

aux_source_directory(traffic_light_utils TRAFFIC_LIGHT_UTILS_SRC)
add_library(traffic_light_utils SHARED ${TRAFFIC_LIGHT_UTILS_SRC} traffic_light_utils/traffic_light_functions.cpp)
target_link_libraries(traffic_light_utils ${OpenCV_LIBS})