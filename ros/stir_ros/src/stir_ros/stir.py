#!/usr/bin/env -S python3 -u
import string

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
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import SetLinkState, SetLinkStateRequest
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, TwistStamped
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
TARGET_POSE = "/pose_tracking_node/target_pose"
SWITCH_CONTROLLER = "/crane_x7/controller_manager/switch_controller"
LIST_CONTROLLERS = "/crane_x7/controller_manager/list_controllers"
NUM_INGREDIENTS = 4  # TODO(sara):get from urdf
NUM_INGREDIENT_POSES = 3  # TODO(sara):get from urdf
INGREDIENT = "ingredient"
INGREDIENTS_MODEL = "ingredient_cube"
LINK = "link"
TOOL_LINK = "tool_link"
BASE_LINK = "base_link"
WORLD = "world"
DEFAULT_CONTROLLER = "arm_controller"
POSITION_CONTROLLER = "arm_position_controller"
VELOCITY_CONTROLLER = "arm_velocity_controller"


class Stir:
    def __init__(self, init_tool_pose: np.ndarray) -> None:
        rospy.init_node("stir_node", anonymous=True)
        self.init_tool_pose = init_tool_pose

        # self._unpause_physics_proxy = rospy.ServiceProxy(UNPAUSE_PHYSICS, Empty)
        # self._unpause_physics_proxy()

        arm_controller_type = rospy.get_param("/arm_controller_type")
        if arm_controller_type == "position":
            self._controller = POSITION_CONTROLLER
        elif arm_controller_type == "velocity":
            self._controller = VELOCITY_CONTROLLER
        else:
            rospy.logerr("invalid controller")

        self._ingredient_buffer = np.zeros(
            (NUM_INGREDIENTS * NUM_INGREDIENT_POSES,)
        )
        self.latest_joint_state = JointState()

        self._obserbation_index = np.array([], dtype=np.uint8)
        topic: LinkStates = rospy.wait_for_message(
            LINK_STATES, LinkStates, timeout=1
        )
        for i, name in enumerate(topic.name):
            if INGREDIENTS_MODEL in name:
                self._obserbation_index = np.append(self._obserbation_index, i)
        if self._obserbation_index.shape != (NUM_INGREDIENTS,):
            rospy.logerr(
                f"NUM_INGREDIENTS(: {NUM_INGREDIENTS}) and \
                number of ingredients in link_states(: {i}) must be equal"
            )

        self._link_states_subscriber = rospy.Subscriber(
            LINK_STATES,
            LinkStates,
            self._link_states_callback,
            queue_size=1,
        )

        self._link_states_subscriber = rospy.Subscriber(
            JOINT_STATES,
            JointState,
            self._joint_states_callback,
            queue_size=1,
        )

        self._target_pose_publishre = rospy.Publisher(
            TARGET_POSE, PoseStamped, queue_size=1
        )

        param_cartesian_command_in_topic = rospy.get_param(
            "pose_tracking_node/cartesian_command_in_topic"
        )
        self._cartesian_command_publisher = rospy.Publisher(
            f"pose_tracking_node/{param_cartesian_command_in_topic}",
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
        self._arm.set_named_target("home")
        self._arm.go(wait=True)
        self._arm.set_named_target("init")
        self._arm.go(wait=True)

        self._switch_controller(self._controller, DEFAULT_CONTROLLER)

        # self._pause_physics_proxy()

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
            self._ingredient_buffer[i * 3] = states.pose[obs_i].position.x
            self._ingredient_buffer[i * 3 + 1] = states.pose[obs_i].position.y
            self._ingredient_buffer[i * 3 + 2] = states.pose[obs_i].position.z

    def _joint_states_callback(self, states: JointState) -> None:
        self.latest_joint_state = states

    def _get_transformed_pose(
        self, target_frame: string, source_frame: string
    ) -> np.ndarray:
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
        return pose

        # cannot use get_current_pose if stop clock

    def get_tool_pose(self) -> np.ndarray:
        return self._get_transformed_pose(WORLD, TOOL_LINK)

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
        self._target_pose_publishre.publish(target_pose_stampd)

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

    def step(self, action: np.ndarray) -> None:
        # TODO(sara): add switcher
        self._publish_twist_stamped(action)
        # self._publish_target_pose(action)

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
        switch_controller.strictness = switch_controller.STRICT
        switch_controller.timeout = 2.0
        self._switch_controller_proxy(switch_controller)

    def reset_robot(self) -> None:
        # self._unpause_physics_proxy()

        if self._get_current_controller() != DEFAULT_CONTROLLER:
            self._switch_controller(DEFAULT_CONTROLLER, self._controller)

        self._arm.set_named_target("init")
        self._arm.go(wait=True)

        rospy.sleep(1)

        self._switch_controller(self._controller, DEFAULT_CONTROLLER)
        # self._pause_physics_proxy()

    def reset_ingredient(self, init_ingredient_poses: np.ndarray) -> None:
        # self._unpause_physics_proxy()
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
        # self._pause_physics_proxy()
