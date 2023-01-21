#include <optional>
#include <thread>

#include <geometry_msgs/TransformStamped.h>
#include <moveit_servo/make_shared_from_pool.h>
#include <moveit_servo/pose_tracking.h>
#include <moveit_servo/servo.h>
#include <moveit_servo/status_codes.h>
#include <ros/ros.h>
#include <std_msgs/Int8.h>

namespace stir_ros {

class PoseTracker {
 public:
  PoseTracker();
  ~PoseTracker();

 private:
  ros::NodeHandle nh_;
  ros::AsyncSpinner spinner_;
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
  std::optional<moveit_servo::PoseTracking> tracker_;
};
}  // namespace stir_ros
