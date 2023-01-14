#pragma once
#include <gazebo/common/Plugin.hh>
#include <ros/ros.h>
#include <std_srvs/Empty.h>

namespace gazebo {

class GazeboBridge : public WorldPlugin {
 public:
  GazeboBridge();

  void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf);

 private:
  bool step_world(std_srvs::Empty::Request& req,
                  std_srvs::Empty::Response& res);
  ros::NodeHandle nh_;
  ros::ServiceServer server_;
  physics::WorldPtr world_;
};

}  // namespace gazebo
