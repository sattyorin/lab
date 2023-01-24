cmake_minimum_required(VERSION 3.0.2)
project(stir_description)

find_package(catkin REQUIRED COMPONENTS gazebo_ros gazebo_plugins
                                        gazebo_ros_control)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

install(DIRECTORY launch worlds DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
