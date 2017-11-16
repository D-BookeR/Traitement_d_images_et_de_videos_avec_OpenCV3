cmake_minimum_required(VERSION 2.8)

PROJECT (ch4KMeans)

find_package(OpenCV  REQUIRED)


file(GLOB ch4KMeans_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE (ch4KMeans ${ch4KMeans_SRCS})



if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	target_link_libraries( ch4KMeans ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

