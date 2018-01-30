cmake_minimum_required(VERSION 2.8)

PROJECT (faciale)

find_package(OpenCV  REQUIRED)


file(GLOB faciale_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE(faciale ${faciale_SRCS})


if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	include_directories( ${EIGEN_INCLUDE_DIRS} )
	target_link_libraries( faciale ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

