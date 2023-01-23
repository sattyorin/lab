#include <ros/ros.h>

#include "bringup_servo.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "bringup_servo_node");
  stir_ros::BringupServo bringup_servo;
  ros::waitForShutdown();
  return EXIT_SUCCESS;
}
