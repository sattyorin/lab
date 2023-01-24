#!/usr/bin/env -S python3 -u
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped

POSITION_CONTROLLER_STIR_ROS_NODE = "pose_tracking_node"
VELOCITY_CONTROLLER_STIR_ROS_NODE = "bringup_servo_node"


class StirRosTest:
    def __init__(self) -> None:
        arm_controller_type = rospy.get_param("/arm_controller_type")
        if arm_controller_type == "position":
            stir_ros_node = POSITION_CONTROLLER_STIR_ROS_NODE
        elif arm_controller_type == "velocity":
            stir_ros_node = VELOCITY_CONTROLLER_STIR_ROS_NODE

        param_cartesian_command_in_topic = rospy.get_param(
            f"{stir_ros_node}/cartesian_command_in_topic"
        )

        self._cartesian_command_publisher = rospy.Publisher(
            f"{stir_ros_node}/{param_cartesian_command_in_topic}",
            TwistStamped,
            queue_size=1,
        )

        self._target_pose_publishre = rospy.Publisher(
            f"{stir_ros_node}/target_pose", PoseStamped, queue_size=1
        )

    def publish_twist_stamped(self, theta: float, max_velocity: float) -> None:
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = rospy.Time.now()
        twist_stamped.header.frame_id = "world"
        twist_stamped.twist.linear.x = max_velocity * np.cos(theta)
        twist_stamped.twist.linear.y = max_velocity * np.sin(theta)
        twist_stamped.twist.linear.z = 0.010
        twist_stamped.twist.angular.x = 0.0
        twist_stamped.twist.angular.y = -0.05
        twist_stamped.twist.angular.z = 0.0
        self._cartesian_command_publisher.publish(twist_stamped)

    def publish_target_pose(self, theta: float, radius: float) -> None:
        target_pose_stampd = PoseStamped()
        target_pose_stampd.header.stamp = rospy.Time.now()
        target_pose_stampd.header.frame_id = "world"
        target_pose_stampd.pose.position.x = radius * np.cos(theta) + 0.375
        target_pose_stampd.pose.position.y = radius * np.sin(theta) - 0.015
        target_pose_stampd.pose.position.z = 0.0 - 0.14
        target_pose_stampd.pose.orientation.x = 0.0 - 0.183
        target_pose_stampd.pose.orientation.y = 0.0 + 0.936
        target_pose_stampd.pose.orientation.z = 0.0 - 0.278
        target_pose_stampd.pose.orientation.w = 0.0 + 0.115
        self._target_pose_publishre.publish(target_pose_stampd)


if __name__ == "__main__":
    rospy.init_node("stir_ros_test_node")
    test = StirRosTest()
    rate = rospy.Rate(100)
    theta = 0.0
    max_velocity = 0.04
    radius = 0.0
    while not rospy.is_shutdown():
        test.publish_twist_stamped(theta, max_velocity)
        # test.publish_target_pose(theta, radius)
        theta += 0.06

        rate.sleep()
