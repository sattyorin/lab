#include "bridge.hpp"

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <ros/ros.h>
#include <std_srvs/Empty.h>

// https://github.com/bhairavmehta95/ros-gazebo-step

namespace gazebo {

constexpr int kTimeStep = 100;

GazeboBridge::GazeboBridge()
    : nh_("~"),
      server_(
          nh_.advertiseService("step_world", &GazeboBridge::step_world, this)) {
  ROS_INFO("ready to GazeboBridge");
};

bool GazeboBridge::step_world(std_srvs::Empty::Request& req,
                              std_srvs::Empty::Response& res) {
  world_->Step(kTimeStep);
  return true;
}

void GazeboBridge::Load(physics::WorldPtr _world, sdf::ElementPtr _sdf) {
  // Make sure the ROS node for Gazebo has already been initialized
  world_ = _world;
  if (!ros::isInitialized()) {
    ROS_FATAL_STREAM(
        "A ROS node for Gazebo has not been initialized, unable to load "
        "plugin. "
        << "Load the Gazebo system plugin 'libgazebo_ros_api_plugin.so' in "
           "the gazebo_ros package)");
    return;
  }

  ROS_INFO("GazeboBridge loaded");
}

GZ_REGISTER_WORLD_PLUGIN(GazeboBridge)
}  // namespace gazebo
