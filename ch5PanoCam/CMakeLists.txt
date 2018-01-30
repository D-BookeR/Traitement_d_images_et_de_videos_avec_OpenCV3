cmake_minimum_required(VERSION 2.8)

PROJECT (ch6PanoCam)

find_package(OpenCV  REQUIRED)


file(GLOB ch6PanoCam_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE(ch6PanoCam ${ch6PanoCam_SRCS})



if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( ch6PanoCam ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)
