#!/usr/bin/env -S python3 -u
import moveit_commander
import numpy as np
import rospy
import tf2_ros
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import SetLinkState, SetLinkStateRequest
from geometry_msgs.msg import Pose, TransformStamped
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty

LINK_STATES = "/gazebo/link_states"
SET_LINK_STATES = "/gazebo/set_link_state"
RESET_SIMULATION = "/gazebo/reset_simulation"
RESET_WORLD = "/gazebo/reset_world"
STEP_WORLD = "/gazebo/step_world"
UNPAUSE_PHYSICS = "gazebo/unpause_physics"
PAUSE_PHYSICS = "gazebo/pause_physics"
NUM_INGREDIENTS = 4  # TODO(sara):get from urdf
NUM_INGREDIENT_POSES = 3  # TODO(sara):get from urdf
INGREDIENT = "ingredient"
INGREDIENTS_MODEL = "ingredient_cube"
LINK = "link"
TOOL_LINK = "tool_link"
BASE_LINK = "base_link"


class Stir:
    def __init__(self, tool_frame: str = TOOL_LINK) -> None:

        self.tool_frame = tool_frame
        self._ingredient_buffer = np.zeros(
            (NUM_INGREDIENTS, NUM_INGREDIENT_POSES)
        )

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

        rospy.wait_for_service(UNPAUSE_PHYSICS, timeout=3)
        self._unpause_physics_proxy = rospy.ServiceProxy(UNPAUSE_PHYSICS, Empty)

        # self._reset_simulation_proxy()
        # rospy.wait_for_message("/clock", Clock)

        self._tfBuffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tfBuffer)
        self._arm = moveit_commander.MoveGroupCommander("arm")

        rospy.sleep(2)
        self._arm.allow_replanning(True)
        rospy.sleep(2)

        self._pause_physics_proxy()

        rospy.loginfo("initalized Stir")

    def _link_states_callback(self, states: LinkStates) -> None:
        for i, obs_i in enumerate(self._obserbation_index):
            self._ingredient_buffer[i, 0] = states.pose[obs_i].position.x
            self._ingredient_buffer[i, 1] = states.pose[obs_i].position.y
            self._ingredient_buffer[i, 2] = states.pose[obs_i].position.z

    def get_tool_pose(self) -> np.ndarray:
        try:
            trans: TransformStamped = self._tfBuffer.lookup_transform(
                BASE_LINK, self.tool_frame, rospy.Duration(0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(
                f"lookup_transform failed: {BASE_LINK} -> {self.tool_frame}\n{e}"
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
        # pose = self._arm.get_current_pose(end_effector_link=BASE_LINK)
        # pose = np.array(
        #     [
        #         pose.pose.position.x,
        #         pose.pose.position.y,
        #         pose.pose.position.z,
        #         pose.pose.orientation.x,
        #         pose.pose.orientation.y,
        #         pose.pose.orientation.z,
        #         pose.pose.orientation.w,
        #     ]
        # )
        # return pose

    def get_ingredient_poses(self) -> np.ndarray:
        return self._ingredient_buffer

    def _move_target_pose(self, pose: np.ndarray, wait=False) -> None:
        target_pose = Pose()
        target_pose.position.x = pose[0]
        target_pose.position.y = pose[1]
        target_pose.position.z = pose[2]
        target_pose.orientation.x = pose[3]
        target_pose.orientation.y = pose[4]
        target_pose.orientation.z = pose[5]
        target_pose.orientation.w = pose[6]
        self._arm.set_pose_target(target_pose, end_effector_link=TOOL_LINK)
        self._arm.go(wait=wait)

    def step(self, action: np.ndarray) -> None:
        self._arm.stop()
        self._move_target_pose(action)
        self._step_world_proxy()

    def reset(
        self, init_tool_pose: np.ndarray, init_ingredient_poses: np.ndarray
    ) -> None:
        self._arm.stop()
        self._unpause_physics_proxy()
        self._move_target_pose(init_tool_pose, wait=True)

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
            link_state.link_state.reference_frame = BASE_LINK
            self._set_link_states_proxy(link_state)

        self._pause_physics_proxy()
