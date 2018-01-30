cmake_minimum_required(VERSION 2.8)

PROJECT (ch3ex1)

find_package(OpenCV  REQUIRED)


file(GLOB ch3ex1_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE(ch3ex1 ${ch3ex1_SRCS})



if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( ch3ex1 ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

