cmake_minimum_required(VERSION 2.8)

PROJECT (ocr_exemple2)

find_package(OpenCV  REQUIRED)

file(GLOB ocr_exemple2_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE (ocr_exemple2 ${ocr_exemple2_SRCS})


if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	include_directories( ${EIGEN_INCLUDE_DIRS} )
	if (VTK_FOUND)
		add_definitions(-DUSE_VTK)
		include_directories( ${VTK_INCLUDE_DIRS} )
	target_link_libraries( facialeBase ${VTK_LIBRARIES} )
	endif (Vtk_FOUND)
	if(ceres_FOUND)
	  add_definitions("-DCERES_FOUND=1")
	endif(ceres_FOUND)
	
	target_link_libraries( ocr_exemple2 ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

