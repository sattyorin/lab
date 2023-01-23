#include "bringup_servo.hpp"

#include <string>
#include <thread>

#include <moveit_servo/make_shared_from_pool.h>
#include <moveit_servo/servo.h>
#include <moveit_servo/status_codes.h>
#include <ros/ros.h>
#include <std_msgs/Int8.h>
#include <std_srvs/Empty.h>

namespace stir_ros {
constexpr int kNumSpinner = 8;

BringupServo::BringupServo()
    : nh_("~"),
      spinner_(kNumSpinner),
      planning_scene_monitor_(
          std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
              "robot_description")) {
  ROS_INFO("initialization BringupServo");
  spinner_.start();

  if (!planning_scene_monitor_->getPlanningScene()) {
    ROS_ERROR_STREAM("Error in setting up the PlanningSceneMonitor.");
    exit(EXIT_FAILURE);
  }
  planning_scene_monitor_->startSceneMonitor();
  planning_scene_monitor_->startWorldGeometryMonitor(
      planning_scene_monitor::PlanningSceneMonitor::
          DEFAULT_COLLISION_OBJECT_TOPIC,
      planning_scene_monitor::PlanningSceneMonitor::
          DEFAULT_PLANNING_SCENE_WORLD_TOPIC,
      false /* skip octomap monitor */);
  planning_scene_monitor_->startStateMonitor();

  servo_.emplace(nh_, planning_scene_monitor_);
  servo_->start();

  ROS_INFO("initialized bringup servo");
}

BringupServo::~BringupServo() {
  // Make sure the tracker is stopped and clean up
  servo_->setPaused(true);
}

}  // namespace stir_ros
