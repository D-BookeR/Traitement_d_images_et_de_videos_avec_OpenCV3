cmake_minimum_required(VERSION 2.8)

PROJECT (ch7Stereo)

find_package(OpenCV  REQUIRED)
find_package(VTK  OPTIONAL_COMPONENTS)


file(GLOB ch7Stereo_SRCS
    "*.h"
    "*.cpp")
ADD_EXECUTABLE (ch7Stereo ${ch7Stereo_SRCS})

if (OpenCV_FOUND)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	if (VTK_FOUND)
		add_definitions(-DUSE_VTK)
		include_directories( ${VTK_INCLUDE_DIRS} )
		target_link_libraries( ch7Stereo ${VTK_LIBRARIES} )
	endif (VTK_FOUND)

	target_link_libraries( ch7Stereo ${OpenCV_LIBS} )
else (OpenCV_FOUND)
message("PB->OPENCV = ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

