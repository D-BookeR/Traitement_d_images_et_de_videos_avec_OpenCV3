cmake_minimum_required(VERSION 2.8)

PROJECT (Ch4canny)

find_package(OpenCV  REQUIRED)


file(GLOB Ch4canny_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE(Ch4canny ${Ch4canny_SRCS})



if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( Ch4canny ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

