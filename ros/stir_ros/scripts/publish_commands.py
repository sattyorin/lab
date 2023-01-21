#!/usr/bin/env -S python3 -u
import rospy
from controller_manager_msgs.srv import SwitchController
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory

SWITCH_CONTROLLER = "/crane_x7/controller_manager/switch_controller"
ARM_CONTROLLER_COMMAND = "/crane_x7/arm_controller/command"
ARM_VELOCITY_CONTOLLER_COMMAND = "/crane_x7/arm_velocity_controller/command"
ORDER = [2, 3, 5, 4, 0, 1, 6]


class PIDTune:
    def __init__(self):
        rospy.wait_for_service(SWITCH_CONTROLLER, timeout=3)
        self._switch_controller_proxy = rospy.ServiceProxy(
            SWITCH_CONTROLLER, SwitchController
        )
        self.arm_controller_publisher = rospy.Publisher(
            ARM_CONTROLLER_COMMAND, JointTrajectory, queue_size=1
        )

        self.arm_velocity_controller_publisher = rospy.Publisher(
            ARM_VELOCITY_CONTOLLER_COMMAND, Float64MultiArray, queue_size=1
        )

    def publish_velocity_command(self, id: int, vel: float) -> None:
        velocity = Float64MultiArray()
        velocity.data = [0.0] * 7
        velocity.data[id] = vel
        self.arm_velocity_controller_publisher.publish(velocity)

    def publish_trajectory_command(self):
        trajectory = JointTrajectory()
        self.arm_controller_publisher.publish(trajectory)


if __name__ == "__main__":
    rospy.init_node("pid_tune_node")
    pid = PIDTune()
    rate = rospy.Rate(40)
    step = 0
    VEL = 0.3
    for id in ORDER:
        step = 0
        while not rospy.is_shutdown():
            if (step // 50) % 2 == 0:
                vel = VEL
            else:
                vel = -VEL
            pid.publish_velocity_command(id, vel)
            step += 1
            rate.sleep()
            if step > 100:
                break
