#include <optional>
#include <thread>

#include <moveit_servo/make_shared_from_pool.h>
#include <moveit_servo/servo.h>
#include <moveit_servo/status_codes.h>
#include <ros/ros.h>
#include <std_msgs/Int8.h>
#include <std_srvs/Empty.h>

namespace stir_ros {

class BringupServo {
 public:
  BringupServo();
  ~BringupServo();

 private:
  bool ResetCallback(std_srvs::Empty::Request& req,
                     std_srvs::Empty::Response& res);
  ros::NodeHandle nh_;
  ros::AsyncSpinner spinner_;
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
  std::optional<moveit_servo::Servo> servo_;
};
}  // namespace stir_ros
