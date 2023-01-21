cmake_minimum_required(VERSION 3.0.2)
project(stir_ros)

set(CMAKE_CXX_STANDARD 17)

find_package(
  catkin REQUIRED
  COMPONENTS rospy
             geometry_msgs
             moveit_commander
             tf2_ros
             gazebo_msgs
             std_srvs
             moveit_servo)

catkin_package(CATKIN_DEPENDS geometry_msgs moveit_commander tf2_ros
               gazebo_msgs std_srvs)

catkin_python_setup()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(pose_tracking_node src/pose_tracker.cpp
                                  src/pose_tracking_node.cpp)
target_link_libraries(pose_tracking_node ${catkin_LIBRARIES})

install(TARGETS pose_tracking_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
