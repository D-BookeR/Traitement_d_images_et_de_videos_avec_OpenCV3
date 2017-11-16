cmake_minimum_required(VERSION 2.8)

PROJECT (ch2ex3)

find_package(OpenCV  REQUIRED)


file(GLOB ch2ex3_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE (ch2ex3 ${ch2ex3_SRCS})



if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
#	include_directories("${OPENCV_CONFIG_PATH}/include")
	target_link_libraries( ch2ex3 ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

