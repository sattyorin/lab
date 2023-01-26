from typing import Tuple

import numpy as np
from envs.stir.i_stir_env import IStirEnv
from envs.stir.stir_util import (
    get_distance_score,
    get_euler_pose_from_quaternion,
)
from gym.spaces import Box


class StirEnv2(IStirEnv):
    def __init__(self, init_tool_pose: np.ndarray):

        is_position_controller = False
        is_velocity_controller = True
        action_is_position = False
        action_is_velocity = True
        observation_tool_pose = np.array(
            [True, True, True, True, True, True, True]
        )
        observation_tool_velocity = np.array(
            [True, True, True, True, True, True]
        )
        observation_ingredient_pose = np.array(
            [False, False, False, False, False, False, False]
        )

        action_low_relative = np.array(
            [-0.10, -0.10, -0.10, *np.deg2rad([-5.0, -5.0, -5.0])]
        )
        action_high_relative = np.array(
            [0.10, 0.10, 0.10, *np.deg2rad([5.0, 5.0, 5.0])]
        )

        init_tool_euler_pose = get_euler_pose_from_quaternion(init_tool_pose)

        action_space = Box(
            low=action_low_relative,
            high=action_high_relative,
            shape=init_tool_euler_pose.shape,
            dtype=np.float32,
        )

        IStirEnv.__init__(
            self,
            is_position_controller,
            is_velocity_controller,
            action_is_position,
            action_is_velocity,
            observation_tool_pose,
            observation_tool_velocity,
            observation_ingredient_pose,
            action_space,
        )

    def _get_controller_input(self, action: np.ndarray) -> np.ndarray:
        return action

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:
        tool_pose = observation[: self._length_tool_pose]
        # tool_velocity = observation[
        #     self._length_tool_pose : self._length_tool_pose
        #     + self._length_tool_velocity
        # ]
        # ingredient_positions = observation[
        #     self._length_tool_pose + self._length_tool_velocity :
        # ]

        reward = get_distance_score(tool_pose[2] - self._init_tool_pose[2])

        if self._init_tool_pose[2] - tool_pose[2] > 0.028:
            return reward, True

        if tool_pose[2] - self._init_tool_pose[2] > 0.02:
            return reward, True

        if self._check_collision_with_bowl(
            self.stir.get_tool_pose()[0], self.stir.get_gripper_pose()[0]
        ):
            return reward, True

        return reward, False
