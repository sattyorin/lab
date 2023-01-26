from typing import Tuple

import envs.stir.stir_util as stir_util
import numpy as np
from envs.stir.i_stir_env import IStirEnv
from gym.spaces import Box

_TARGET_VELOCITY = 0.08
_THRESHOLD_DISTANCE = 0.01


class StirEnv1(IStirEnv):
    def __init__(self, init_tool_pose: np.ndarray):

        is_position_controller = True
        is_velocity_controller = False
        action_is_position = True
        action_is_velocity = False
        observation_tool_pose = np.array(
            [True, True, False, False, False, False, False]
        )
        observation_tool_velocity = np.array(
            [True, True, False, False, False, False]
        )
        observation_ingredient_pose = np.array(
            [True, True, False, False, False, False, False]
        )

        action_low = np.array([0.0, 0.0])
        action_high = np.array([0.08, 0.1])

        # init_tool_euler_pose = stir_util.get_euler_pose_from_quaternion(
        #     init_tool_pose
        # )

        action_space = Box(
            low=action_low,
            high=action_high,
            shape=action_low.shape,
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
        control = action.copy()
        control[0] = action[0] * np.cos(self._previous_angle + action[1])
        control[1] = action[0] * np.sin(self._previous_angle + action[1])
        self._previous_angle += action[1]
        return control

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:
        # tool_pose = observation[:self._length_tool_pose]
        tool_velocity = observation[
            self._length_tool_pose : self._length_tool_pose
            + self._length_tool_velocity
        ]
        ingredient_positions = observation[
            self._length_tool_pose + self._length_tool_velocity :
        ]

        reward_small_velocity = stir_util.get_reward_small_velocity(
            np.linalg.norm(tool_velocity[:2]), _TARGET_VELOCITY
        )
        self._total_velocity_reward += reward_small_velocity

        distance = stir_util.get_distance_between_two_centroid(
            ingredient_positions.reshape(
                -1, self._dimension_ingredient_distance
            ),
            self._every_other_ingredients,
        )
        reward_distance = stir_util.get_reward_stir(distance)

        reward = reward_small_velocity + reward_distance

        # if self._check_collision_with_bowl():
        #     return -100, True
        # if self._total_velocity_reward / (self.num_step + 1) < 0.2:
        #     print(self._total_velocity_reward)
        #     return reward, True

        # if distance < _THRESHOLD_DISTANCE:
        #     print("done")
        #     return 100, True

        return reward, False
