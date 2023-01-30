import os
from typing import Tuple

import envs.stir.stir_util as stir_util
import numpy as np
import yaml
from envs.stir.i_stir_env import IStirEnv
from gym.spaces import Box

_TARGET_VELOCITY = 0.02
_THRESHOLD_DISTANCE = 0.01


class StirEnvXYZTouchIngredients(IStirEnv):
    def __init__(self, init_tool_pose: np.ndarray):

        is_position_controller = True
        is_velocity_controller = False
        action_is_position = False
        action_is_velocity = True
        observation_tool_pose = np.array(
            [True, True, True, False, False, False, False]
        )
        observation_tool_velocity = np.array(
            [True, True, True, False, False, False]
        )
        observation_ingredient_pose = np.array(
            [True, True, False, False, False, False, False]
        )

        action_low = np.array([-1.0, -1.0, 0.0])
        action_high = np.array([1.0, 1.0, 0.013])  # ingredient_height

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

        # get config
        env_directory = os.path.abspath(os.path.dirname(__file__))
        bowl_param_path = os.path.join(
            env_directory, "config", "bowl_param.yaml"
        )
        with open(bowl_param_path, "r") as f:
            self._bowl = yaml.safe_load(f)
            tools_radius = 0.008
            for key in self._bowl:
                self._bowl[key] -= tools_radius

    def _get_controller_input(self, action: np.ndarray) -> np.ndarray:
        a = (
            self._bowl["bowl_radius_top"] - self._bowl["bowl_radius_bottom"]
        ) / self._bowl["bowl_height"] * action[2] + self._bowl[
            "bowl_radius_bottom"
        ]

        return np.array(
            [
                a * action[0],
                a * action[1],
                action[2] - self._bowl["bowl_bottom_to_init_z"],
            ]
        )  # for mujoco?

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:
        tool_pose = observation[: self._length_tool_pose]
        # tool_velocity = observation[
        #     self._length_tool_pose : self._length_tool_pose
        #     + self._length_tool_velocity
        # ]
        ingredient_positions = observation[
            self._length_tool_pose + self._length_tool_velocity :
        ]

        if self._previous_ingredient_positions is not None:
            reward = np.exp(
                -np.linalg.norm((ingredient_positions[:2] - tool_pose[:2]) * 20)
            )

        else:
            reward = 0.0

        if (
            self._previous_ingredient_positions is not None
            and np.linalg.norm(
                ingredient_positions[: self._dimension_ingredient_distance]
                - self._previous_ingredient_positions[
                    : self._dimension_ingredient_distance
                ]
            )
            > 0.0000001
        ):
            return 500, True

        # self._total_velocity_reward += stir_util.get_reward_small_velocity(
        #     np.linalg.norm(tool_velocity[:3]), _TARGET_VELOCITY
        # )

        # if self._total_velocity_reward / (self.num_step + 1) < 0.5:
        #     print(self._total_velocity_reward)
        #     return reward, True

        self._previous_ingredient_positions = ingredient_positions
        return reward, False
