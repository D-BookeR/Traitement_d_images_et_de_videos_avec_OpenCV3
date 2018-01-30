cmake_minimum_required(VERSION 2.8)

PROJECT (chSfm)

find_package(glog QUIET REQUIRED)
find_package(ceres QUIET REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(VTK  OPTIONAL_COMPONENTS)


file(GLOB chSfm_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE(chSfm ${chSfm_SRCS})


if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	include_directories( ${EIGEN_INCLUDE_DIRS} )
	if (VTK_FOUND)
		add_definitions(-DUSE_VTK)
		include_directories( ${VTK_INCLUDE_DIRS} )
		target_link_libraries( chSfm ${VTK_LIBRARIES} )
	endif (Vtk_FOUND)
	if(ceres_FOUND)
	    add_definitions("-DCERES_FOUND=1")
	endif(ceres_FOUND)
	target_link_libraries( chSfm ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

