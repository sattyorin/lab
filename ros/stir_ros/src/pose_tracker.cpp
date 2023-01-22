#include "pose_tracker.hpp"

#include <string>
#include <thread>

#include <geometry_msgs/TransformStamped.h>
#include <moveit_servo/make_shared_from_pool.h>
#include <moveit_servo/pose_tracking.h>
#include <moveit_servo/servo.h>
#include <moveit_servo/status_codes.h>
#include <ros/ros.h>
#include <std_msgs/Int8.h>
#include <std_srvs/Empty.h>

namespace stir_ros {
constexpr int kNumSpinner = 8;
const std::string kPoseTrackerTopic = "pose_tracker";
constexpr double kAngularTolerance = 0.01;
const Eigen::Vector3d kPositionalTolerance{0.001, 0.001, 0.001};
constexpr double kTargetPoseTimeout = 100.0;

PoseTracker::PoseTracker()
    : nh_("~"),
      spinner_(kNumSpinner),
      reset_subscriber_(nh_.advertiseService(
          "reset_moveit_servo", &PoseTracker::ResetCallback, this)),
      planning_scene_monitor_(
          std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
              "robot_description")) {
  ROS_INFO("initialization PoseTracker");
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

  tracker_.emplace(nh_, planning_scene_monitor_);

  tracker_->resetTargetPose();

  while (ros::ok()) {
    tracker_->moveToPose(kPositionalTolerance, kAngularTolerance,
                         kTargetPoseTimeout);
  }
  ROS_INFO("initialized pose tracker");
}

PoseTracker::~PoseTracker() {
  // Make sure the tracker is stopped and clean up
  tracker_->stopMotion();
}

bool PoseTracker::ResetCallback(std_srvs::Empty::Request& req,
                                std_srvs::Empty::Response& res) {
  tracker_->resetTargetPose();
  return true;
}

}  // namespace stir_ros
