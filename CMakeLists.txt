cmake_minimum_required(VERSION 3.0.2)
project(p2i_cast)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  ros_numpy
  std_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
  image_geometry
  image_transport
  image_view
  OpenCV
  tf
)

catkin_package(CATKIN_DEPENDS
	sensor_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

find_package(Eigen3 REQUIRED)

#SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=c++17 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_STANDARD 17) #14
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

#Ground Plane Fitter
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

catkin_install_python(PROGRAMS p2i_cast_sp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(p2i_cast p2i_cast.cpp)

target_include_directories(p2i_cast PRIVATE
	${PCL_INCLUDE_DIRS}
)

target_link_libraries(p2i_cast
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
)
