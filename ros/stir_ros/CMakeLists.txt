cmake_minimum_required(VERSION 3.0.2)
project(stir_ros)

find_package(catkin REQUIRED COMPONENTS rospy geometry_msgs moveit_commander
                                        tf2_ros gazebo_msgs std_srvs)

catkin_package(CATKIN_DEPENDS geometry_msgs moveit_commander tf2_ros
               gazebo_msgs std_srvs)

catkin_python_setup()

include_directories(${catkin_INCLUDE_DIRS})

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
