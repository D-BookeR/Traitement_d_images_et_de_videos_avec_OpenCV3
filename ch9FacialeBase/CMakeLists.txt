cmake_minimum_required(VERSION 2.8)

PROJECT (facialeBase)

find_package(OpenCV  REQUIRED)


file(GLOB facialeBase_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE (facialeBase ${facialeBase_SRCS})


if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( facialeBase ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

