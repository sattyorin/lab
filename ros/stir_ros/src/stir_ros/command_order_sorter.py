#!/usr/bin/env -S python3 -u
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

SORT_ORDER = np.array(
    [
        4,  # crane_x7_lower_arm_fixed_part_joint
        5,  # crane_x7_lower_arm_revolute_part_joint
        0,  # crane_x7_shoulder_fixed_part_pan_joint
        1,  # crane_x7_shoulder_revolute_part_tilt_joint
        3,  # crane_x7_upper_arm_revolute_part_rotate_joint
        2,  # crane_x7_upper_arm_revolute_part_twist_joint
        6,  # crane_x7_wrist_joint
    ]
)

JOINT_STATES = "/crane_x7/joint_states"
ALPHA = 0.3


class Sorter:
    def __init__(self) -> None:
        param_command_out_type = rospy.get_param("command_out_type")
        if param_command_out_type == "std_msgs/Float64MultiArray":
            command_out_type = Float64MultiArray
        else:
            rospy.logerr("rosparam command_out_type is invalid")

        param_controller_command_topic = rospy.get_param(
            "controller_command_topic"
        )

        param_command_out_topic = rospy.get_param("command_out_topic")

        param_publish_joint_positions = rospy.get_param(
            "publish_joint_positions"
        )
        param_publish_joint_velocities = rospy.get_param(
            "publish_joint_velocities"
        )
        param_arm_controller_type = rospy.get_param("/arm_controller_type")

        self._param_publish_period = float(rospy.get_param("publish_period"))

        rospy.loginfo(
            f"param_controller_command_topic: {param_controller_command_topic}"
        )

        if param_publish_joint_positions:
            callback = self._callback_position
        elif (
            param_publish_joint_velocities
            and param_arm_controller_type == "position"
        ):
            callback = self._callback_position_from_velocity
        elif (
            param_publish_joint_velocities
            and param_arm_controller_type == "velocity"
        ):
            callback = self._callback_velocity
        else:
            rospy.logerr("set publish joint position or state to true")

        callback = self._callback_velocity

        self._latest_joint_state = np.array(
            rospy.wait_for_message(JOINT_STATES, JointState).position
        )[1:]

        self.previous_joint_position_command = self._latest_joint_state

        self._publisher = rospy.Publisher(
            param_controller_command_topic,
            command_out_type,
            queue_size=10,
        )

        self._link_states_subscriber = rospy.Subscriber(
            JOINT_STATES,
            JointState,
            self._joint_states_callback,
            queue_size=1,
        )

        rospy.wait_for_message(JOINT_STATES, JointState)

        self._subscriber = rospy.Subscriber(
            param_command_out_topic,
            command_out_type,
            callback,
            queue_size=1,
        )

    def _joint_states_callback(self, states: JointState) -> None:
        # self._latest_joint_state = np.array(states.position[1:])
        self._latest_joint_state += (
            np.array(states.position[1:]) - self._latest_joint_state
        ) * ALPHA

    def _callback_position(self, msg: Float64MultiArray) -> None:
        new_msg = Float64MultiArray()
        new_msg.layout = msg.layout
        new_msg.data = [msg.data[id] for id in SORT_ORDER]
        self._publisher.publish(new_msg)

    def _callback_position_from_velocity(self, msg: Float64MultiArray) -> None:
        new_position_command = np.array(msg.data)[SORT_ORDER]
        # filltered_joint_position = (
        #     ALPHA * self._latest_joint_state
        #     + (1 - ALPHA) * self.previous_joint_position_command
        # )
        new_position_command = (
            new_position_command * self._param_publish_period * 5.0
            + self._latest_joint_state
            # + filltered_joint_position
        )
        new_msg = Float64MultiArray()
        new_msg.layout = msg.layout
        new_msg.data = new_position_command.tolist()
        self._publisher.publish(new_msg)
        self.previous_joint_position_command = new_position_command

    def _callback_velocity(self, msg: Float64MultiArray) -> None:
        new_msg = Float64MultiArray()
        new_msg.layout = msg.layout
        new_msg.data = [msg.data[id] for id in SORT_ORDER]
        self._publisher.publish(new_msg)


if __name__ == "__main__":
    rospy.init_node("sorter_node")
    sorter = Sorter()
    rospy.loginfo("sorter_node started")
    rospy.spin()
