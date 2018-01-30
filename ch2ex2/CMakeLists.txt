cmake_minimum_required(VERSION 2.8)

PROJECT (ch2ex2)

find_package(OpenCV  REQUIRED)

file(GLOB ch2ex2_SRCS
    "*.h"
    "*.cpp")

ADD_EXECUTABLE(ch2ex2 ${ch2ex2_SRCS})

if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( ch2ex2 ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

