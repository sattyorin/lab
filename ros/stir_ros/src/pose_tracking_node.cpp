#include <ros/ros.h>

#include "pose_tracker.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "pose_tracking_node");
  stir_ros::PoseTracker pose_tracker;
  ros::waitForShutdown();
  return EXIT_SUCCESS;
}
