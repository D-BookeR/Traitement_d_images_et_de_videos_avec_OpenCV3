cmake_minimum_required(VERSION 2.8)

PROJECT (ch4Watershed)

find_package(OpenCV  REQUIRED)


file(GLOB ch4Watershed_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE (ch4Watershed ${ch4Watershed_SRCS})



if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( ch4Watershed ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

