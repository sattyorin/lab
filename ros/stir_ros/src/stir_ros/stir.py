#!/usr/bin/env -S python3 -u
import string
from typing import Tuple

import moveit_commander
import numpy as np
import rospy
import tf2_ros
from controller_manager_msgs.srv import (
    ListControllers,
    ListControllersResponse,
    SwitchController,
    SwitchControllerRequest,
)
from gazebo_msgs.msg import LinkStates, ModelStates
from gazebo_msgs.srv import SetLinkState, SetLinkStateRequest
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, TwistStamped
from moveit_msgs.msg import MoveGroupActionResult
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty

# from rosgraph_msgs.msg import Clock

LINK_STATES = "/gazebo/link_states"
SET_LINK_STATES = "/gazebo/set_link_state"
RESET_SIMULATION = "/gazebo/reset_simulation"
RESET_WORLD = "/gazebo/reset_world"
STEP_WORLD = "/gazebo/step_world"
UNPAUSE_PHYSICS = "/gazebo/unpause_physics"
PAUSE_PHYSICS = "/gazebo/pause_physics"
JOINT_STATES = "/crane_x7/joint_states"
POSITION_CONTROLLER_STIR_ROS_NODE = "pose_tracking_node"
VELOCITY_CONTROLLER_STIR_ROS_NODE = "bringup_servo_node"
SWITCH_CONTROLLER = "/crane_x7/controller_manager/switch_controller"
LIST_CONTROLLERS = "/crane_x7/controller_manager/list_controllers"
NUM_INGREDIENT_POSES = 7  # TODO(sara):get from urdf
INGREDIENT = "ingredient"
INGREDIENTS_MODEL = "ingredient_cube"
LINK = "link"
TOOL_LINK = "tool_link"
BASE_LINK = "base_link"
WORLD = "world"
GRIPPER_BASE_LINK = "crane_x7_gripper_base_link"
DEFAULT_CONTROLLER = "arm_controller"
POSITION_CONTROLLER = "arm_position_controller"
VELOCITY_CONTROLLER = "arm_velocity_controller"
_RESET_NOISE_SCALE = 0.001


class Stir:
    def __init__(self, init_tool_pose: np.ndarray) -> None:
        rospy.init_node("stir_node", anonymous=True)
        self.init_tool_pose = init_tool_pose
        self._moveit_error = False

        # self._unpause_physics_proxy = rospy.ServiceProxy(UNPAUSE_PHYSICS, Empty)
        # self._unpause_physics_proxy()

        arm_controller_type = rospy.get_param("/arm_controller_type")
        if arm_controller_type == "position":
            self._controller = POSITION_CONTROLLER
            stir_ros_node = POSITION_CONTROLLER_STIR_ROS_NODE
        elif arm_controller_type == "velocity":
            self._controller = VELOCITY_CONTROLLER
            stir_ros_node = VELOCITY_CONTROLLER_STIR_ROS_NODE
        else:
            rospy.logerr("invalid controller")

        model_states: ModelStates = rospy.wait_for_message(
            "/gazebo/model_states", ModelStates, timeout=2
        )
        self.num_ingredients = 0
        for name in model_states.name:
            if "ingredient" in name:
                self.num_ingredients += 1

        self._ingredient_buffer = np.zeros(
            (self.num_ingredients * NUM_INGREDIENT_POSES,)
        )
        self.latest_joint_state = JointState()

        self._obserbation_index = np.array([], dtype=np.uint8)
        topic: LinkStates = rospy.wait_for_message(
            LINK_STATES, LinkStates, timeout=1
        )
        for i, name in enumerate(topic.name):
            if INGREDIENTS_MODEL in name:
                self._obserbation_index = np.append(self._obserbation_index, i)
        if self._obserbation_index.shape != (self.num_ingredients,):
            rospy.logerr(
                f"NUM_INGREDIENTS(: {self.num_ingredients}) and \
                number of ingredients in link_states(: {i}) must be equal"
            )

        self._link_states_subscriber = rospy.Subscriber(
            LINK_STATES,
            LinkStates,
            self._link_states_callback,
            queue_size=1,
        )

        self._joint_states_subscriber = rospy.Subscriber(
            JOINT_STATES,
            JointState,
            self._joint_states_callback,
            queue_size=1,
        )

        self._move_group_result_subscriber = rospy.Subscriber(
            "/move_group/result",
            MoveGroupActionResult,
            self._move_group_result_callback,
            queue_size=1,
        )

        self._target_pose_publisher = rospy.Publisher(
            f"/{stir_ros_node}/target_pose", PoseStamped, queue_size=1
        )

        param_cartesian_command_in_topic = rospy.get_param(
            f"/{stir_ros_node}/cartesian_command_in_topic"
        )
        self._cartesian_command_publisher = rospy.Publisher(
            f"/{stir_ros_node}/{param_cartesian_command_in_topic}",
            TwistStamped,
            queue_size=1,
        )

        rospy.wait_for_service(SET_LINK_STATES, timeout=3)
        self._set_link_states_proxy = rospy.ServiceProxy(
            SET_LINK_STATES, SetLinkState
        )

        rospy.wait_for_service(RESET_SIMULATION, timeout=3)
        self._reset_simulation_proxy = rospy.ServiceProxy(
            RESET_SIMULATION, Empty
        )

        rospy.wait_for_service(RESET_WORLD, timeout=3)
        self._reset_world_proxy = rospy.ServiceProxy(RESET_WORLD, Empty)

        rospy.wait_for_service(STEP_WORLD, timeout=3)
        self._step_world_proxy = rospy.ServiceProxy(STEP_WORLD, Empty)

        rospy.wait_for_service(PAUSE_PHYSICS, timeout=3)
        self._pause_physics_proxy = rospy.ServiceProxy(PAUSE_PHYSICS, Empty)

        rospy.wait_for_service(SWITCH_CONTROLLER, timeout=3)
        self._switch_controller_proxy = rospy.ServiceProxy(
            SWITCH_CONTROLLER, SwitchController
        )

        rospy.wait_for_service(LIST_CONTROLLERS, timeout=3)
        self._list_controllers_proxy = rospy.ServiceProxy(
            LIST_CONTROLLERS, ListControllers
        )

        rospy.wait_for_service(UNPAUSE_PHYSICS, timeout=2)
        self._unpause_physics_proxy = rospy.ServiceProxy(UNPAUSE_PHYSICS, Empty)

        # self._reset_simulation_proxy()
        # rospy.wait_for_message("/clock", Clock)

        self._tfBuffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tfBuffer)

        if self._get_current_controller() != DEFAULT_CONTROLLER:
            self._switch_controller(DEFAULT_CONTROLLER, self._controller)

        self._arm = moveit_commander.MoveGroupCommander("arm")
        self._gripper = moveit_commander.MoveGroupCommander("gripper")
        self._robot = moveit_commander.RobotCommander()
        self._arm.set_end_effector_link(TOOL_LINK)
        # self._arm.allow_replanning(True)
        # self._arm.set_planning_time(0.2)

        print("========================================")
        print(f"get_end_effector_link: {self._arm.get_end_effector_link()}")
        print(
            f"get_pose_reference_frame: {self._arm.get_pose_reference_frame()}"
        )
        print(
            f"get_goal_joint_tolerance: {self._arm.get_goal_joint_tolerance()}"
        )
        print(
            f"get_goal_orientation_tolerance: {self._arm.get_goal_orientation_tolerance()}"
        )
        print(
            f"get_goal_position_tolerance: {self._arm.get_goal_position_tolerance()}"
        )
        print(f"get_root_link: {self._robot.get_root_link()}")
        print("========================================")

        rospy.loginfo("go init pose")
        self._arm.set_named_target("pre_init")
        self._arm.go(wait=True)
        self._arm.set_named_target("init")
        self._arm.go(wait=True)

        if self._moveit_error:
            rospy.logerr("failed to go init pose")

        self._switch_controller(self._controller, DEFAULT_CONTROLLER)

        self._pause_physics_proxy()

        rospy.loginfo("initalized Stir")

    def _get_current_controller(self) -> string:
        controllers: ListControllersResponse = self._list_controllers_proxy()
        for controller in controllers.controller:
            if controller.state == "running":
                if controller.name in {DEFAULT_CONTROLLER, self._controller}:
                    return controller.name

        rospy.signal_shutdown("unknown controller")

    def _get_pose_from_array(self, array: np.ndarray) -> Pose:
        pose = Pose()
        pose.position.x = array[0]
        pose.position.y = array[1]
        pose.position.z = array[2]
        pose.orientation.x = array[3]
        pose.orientation.y = array[4]
        pose.orientation.z = array[5]
        pose.orientation.w = array[6]
        return pose

    def _link_states_callback(self, states: LinkStates) -> None:
        for i, obs_i in enumerate(self._obserbation_index):
            self._ingredient_buffer[i * 7] = states.pose[obs_i].position.x
            self._ingredient_buffer[i * 7 + 1] = states.pose[obs_i].position.y
            self._ingredient_buffer[i * 7 + 2] = states.pose[obs_i].position.z
            self._ingredient_buffer[i * 7 + 3] = states.pose[
                obs_i
            ].orientation.x
            self._ingredient_buffer[i * 7 + 4] = states.pose[
                obs_i
            ].orientation.y
            self._ingredient_buffer[i * 7 + 5] = states.pose[
                obs_i
            ].orientation.z
            self._ingredient_buffer[i * 7 + 6] = states.pose[
                obs_i
            ].orientation.w

    def _joint_states_callback(self, states: JointState) -> None:
        self.latest_joint_state = states

    def _move_group_result_callback(self, result: MoveGroupActionResult):
        self._moveit_error = result.result.error_code.val < 0

    def _get_transformed_pose(
        self,
        target_frame: string,
        source_frame: string,
    ) -> Tuple[np.ndarray, rospy.Time]:
        try:
            trans: TransformStamped = self._tfBuffer.lookup_transform(
                target_frame, source_frame, rospy.Duration(0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(
                f"lookup_transform failed: {target_frame} -> {source_frame}\n{e}"
            )

        pose = np.array(
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z,
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            ]
        )
        return pose, trans.header.stamp

        # cannot use get_current_pose if stop clock

    def get_tool_pose(self) -> Tuple[np.ndarray, float]:
        pose, stamp = self._get_transformed_pose(WORLD, TOOL_LINK)
        return pose, stamp.to_sec()

    def get_gripper_pose(self) -> Tuple[np.ndarray, float]:
        pose, stamp = self._get_transformed_pose(WORLD, GRIPPER_BASE_LINK)
        return pose, stamp.to_sec()

    def get_current_state(self) -> JointState:
        return self.latest_joint_state
        # cannot use get_current_state if stop clock

    def get_ingredient_poses(self) -> np.ndarray:
        return self._ingredient_buffer

    def _publish_target_pose(self, pose: np.ndarray, wait=False) -> None:
        target_pose = self._get_pose_from_array(pose)
        target_pose_stampd = PoseStamped()
        target_pose_stampd.header.stamp = rospy.Time.now()
        target_pose_stampd.header.frame_id = WORLD
        target_pose_stampd.pose = target_pose
        self._target_pose_publisher.publish(target_pose_stampd)

    def _publish_twist_stamped(self, twist: np.ndarray) -> None:
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = rospy.Time.now()
        twist_stamped.header.frame_id = WORLD
        twist_stamped.twist.linear.x = twist[0]
        twist_stamped.twist.linear.y = twist[1]
        twist_stamped.twist.linear.z = twist[2]
        twist_stamped.twist.angular.x = twist[3]
        twist_stamped.twist.angular.y = twist[4]
        twist_stamped.twist.angular.z = twist[5]
        self._cartesian_command_publisher.publish(twist_stamped)

    def step_position_controller(self, action: np.ndarray) -> None:
        self._publish_target_pose(action)
        self._step_world_proxy()

    def step_velocity_controller(self, action: np.ndarray) -> None:
        self._publish_twist_stamped(action)
        self._step_world_proxy()

    def _switch_controller(
        self, start_controller: string, stop_controller: string
    ) -> None:
        for controller in [start_controller, stop_controller]:
            if controller not in [
                self._controller,
                DEFAULT_CONTROLLER,
            ]:
                rospy.logerr(f"controller({controller}) dose not exist")
        switch_controller = SwitchControllerRequest()
        switch_controller.start_controllers.append(start_controller)
        switch_controller.stop_controllers.append(stop_controller)
        switch_controller.strictness = SwitchControllerRequest.STRICT
        switch_controller.timeout = 2.0
        self._switch_controller_proxy(switch_controller)

    def reset_robot(self) -> None:
        self._unpause_physics_proxy()

        if self._get_current_controller() != DEFAULT_CONTROLLER:
            self._switch_controller(DEFAULT_CONTROLLER, self._controller)

        self._arm.set_named_target("pre_init")
        self._arm.go(wait=True)
        self._arm.set_named_target("init")
        self._arm.go(wait=True)
        while self._moveit_error:
            self._reset_world_proxy()
            # TODO(sara): set_link_state ?
            self._arm.set_named_target("pre_init")
            self._arm.go(wait=True)
            self._arm.set_named_target("init")
            self._arm.go(wait=True)

        joint_values = np.array(self._arm.get_current_joint_values())
        joint_values += np.random.uniform(
            low=-_RESET_NOISE_SCALE,
            high=_RESET_NOISE_SCALE,
            size=7,
        )
        self._arm.set_joint_value_target(joint_values)

        rospy.sleep(1)

        self._switch_controller(self._controller, DEFAULT_CONTROLLER)
        self._pause_physics_proxy()

    def reset_ingredient(self, init_ingredient_poses: np.ndarray) -> None:
        self._unpause_physics_proxy()
        self._reset_world_proxy()

        for i, ingredient_pose in enumerate(init_ingredient_poses):
            link_state = SetLinkStateRequest()
            link_state.link_state.link_name = (
                f"{INGREDIENT}{i}::{INGREDIENTS_MODEL}::{LINK}"
            )
            link_state.link_state.pose.position.x = ingredient_pose[0]
            link_state.link_state.pose.position.y = ingredient_pose[1]
            link_state.link_state.pose.position.z = ingredient_pose[2]
            link_state.link_state.pose.orientation.x = 0.0
            link_state.link_state.pose.orientation.y = 0.0
            link_state.link_state.pose.orientation.z = 0.0
            link_state.link_state.pose.orientation.w = 1.0
            link_state.link_state.reference_frame = WORLD
            self._set_link_states_proxy(link_state)

        rospy.wait_for_message(LINK_STATES, LinkStates, timeout=2)
        self._pause_physics_proxy()
