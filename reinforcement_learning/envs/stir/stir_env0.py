import os
from typing import Optional, Tuple

import envs.stir.stir_utils as stir_utils
import numpy as np
import yaml
from envs.stir.i_stir_env import IStirEnv
from gym.spaces import Box

_TARGET_VELOCITY = 0.03


class StirEnv0(IStirEnv):
    def __init__(
        self,
        init_tool_pose: np.ndarray,
        num_ingredients: int,
        check_collision_with_bowl,
    ):
        self._init_tool_pose = init_tool_pose
        self._num_ingredients = num_ingredients
        self._check_collision_with_bowl = check_collision_with_bowl

        is_position_controller = True
        is_velocity_controller = False
        action_is_position = True
        action_is_velocity = False
        observation_tool_pose = np.array(
            [True, True, True, False, False, False, False]
        )
        observation_tool_velocity = np.array(
            [True, True, True, False, False, False]
        )
        observation_ingredient_pose = np.array(
            [True, True, False, False, False, False, False]
        )

        # action_low_relative = np.array(
        #     [-0.30, -0.30, -0.2, -45.0, -45.0, -45.0]
        # )
        # action_high_relative = np.array([0.30, 0.30, 0.20, 45.0, 45.0, 45.0])

        # init_tool_euler_pose = stir_utils.get_euler_pose_from_quaternion(
        #     init_tool_pose
        # )

        # action_low = action_low_relative + init_tool_euler_pose
        # action_high = action_high_relative + init_tool_euler_pose

        action_low = np.array([-1.0, -1.0, 0.0])
        action_high = np.array([1.0, 1.0, 0.014])  # ingredient_height

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
            num_ingredients,
        )

        env_directory = os.path.abspath(os.path.dirname(__file__))
        tool_param_path = os.path.join(
            env_directory, "config", "tool_param.yaml"
        )
        bowl_param_path = os.path.join(
            env_directory, "config", "bowl_param.yaml"
        )
        with open(tool_param_path, "r") as f:
            tool = yaml.safe_load(f)
        with open(bowl_param_path, "r") as f:
            self._bowl = yaml.safe_load(f)
            for key in self._bowl:
                self._bowl[key] -= tool["radius_smallest_circle"]

        self._every_other_ingredients = False
        self._total_reward_keep_moving_end_effector = 0.0
        self._num_step = 0
        self._previous_ingredient_positions: Optional[np.ndarray] = None
        self._total_reward_array_keep_moving_ingredients = np.zeros(
            self._num_ingredients
        )

    def get_controller_input(self, action: np.ndarray) -> np.ndarray:
        a = (
            self._bowl["radius_top"] - self._bowl["radius_bottom"]
        ) / self._bowl["height"] * action[2] + self._bowl["radius_bottom"]

        return np.array(
            [
                a * action[0],
                a * action[1],
                action[2] - self._bowl["bottom_to_init_z"],
            ]
        )  # for mujoco?

    def get_reward(
        self, observation: np.ndarray, reset_mode: bool = False
    ) -> Tuple[float, bool]:
        tool_pose = observation[: self._length_tool_pose]
        tool_velocity = observation[
            self._length_tool_pose : self._length_tool_pose
            + self._length_tool_velocity
        ]
        ingredient_positions = observation[
            self._length_tool_pose + self._length_tool_velocity :
        ]

        # -------------------------------------------------------
        reward_keep_moving_end_effector = (
            stir_utils.get_reward_keep_moving_end_effector(
                np.linalg.norm(tool_velocity[:2]), _TARGET_VELOCITY
            )
        )
        self._total_reward_keep_moving_end_effector += (
            reward_keep_moving_end_effector
        )

        if (
            self._total_reward_keep_moving_end_effector / (self._num_step + 1)
            < 0.2
        ):
            return -1000.0, True
        # -------------------------------------------------------

        # -------------------------------------------------------
        reward_distance_between_tool_and_ingredient = (
            stir_utils.get_reward_distance_between_tool_and_ingredient(
                ingredient_positions[:2],
                tool_pose[:2],
                self._bowl["radius_bottom"],
            )
        )

        if self.detect_touch(ingredient_positions):
            return 1000.0, True
        # -------------------------------------------------------

        # -------------------------------------------------------
        if self._previous_ingredient_positions is not None:
            reward_array_keep_moving_ingredients = np.round(
                stir_utils.get_reward_array_keep_moving_ingredients(
                    ingredient_positions.reshape(
                        -1, self._length_ingredient_pose
                    )[:, : self._dimension_ingredient_distance],
                    self._previous_ingredient_positions.reshape(
                        -1, self._length_ingredient_pose
                    )[:, : self._dimension_ingredient_distance],
                    self._bowl["radius_bottom"] * 0.02,
                ),
                3,
            )
            self._total_reward_array_keep_moving_ingredients += (
                reward_array_keep_moving_ingredients
            )
            reward_keep_moving_ingredients = np.mean(
                reward_array_keep_moving_ingredients
            )
        else:
            reward_keep_moving_ingredients = 0.0

        if (
            sum(
                self._total_reward_array_keep_moving_ingredients
                / (self._num_step + 1)
                > 0.03
            )
            < 2
        ):
            return -1000.0, True
        # -------------------------------------------------------

        # -------------------------------------------------------
        distance_array_to_center_point = (
            stir_utils.get_distance_array_to_center_point(
                ingredient_positions.reshape(-1, self._length_ingredient_pose)[
                    :, :2
                ]
            )
        )
        reward_array_distance_to_center_pointrd = (
            stir_utils.get_reward_array_distance_to_center_point(
                distance_array_to_center_point, self._bowl["radius_top"]
            )
        )
        reward_distance_to_center_pointrd = np.mean(
            reward_array_distance_to_center_pointrd
        )
        if (reward_array_distance_to_center_pointrd > 0.9).all():
            return 1000.0, True
        # -------------------------------------------------------

        # -------------------------------------------------------
        distance_to_center_plane = stir_utils.get_distance_to_center_plane(
            ingredient_positions.reshape(-1, self._length_ingredient_pose)[
                :, : self._dimension_ingredient_distance
            ],
            self._every_other_ingredients,
        )
        reward_distance_to_center_plane = (
            stir_utils.get_reward_distance_to_center_plane(
                distance_to_center_plane,
                self._bowl["radius_bottom"] * self._num_ingredients,
            )
        )

        if reward_distance_to_center_plane > 0.9:
            return 1000.0, True
        # -------------------------------------------------------

        # -------------------------------------------------------
        distance_between_two_centroids = (
            stir_utils.get_distance_between_two_centroids(
                ingredient_positions.reshape(-1, self._length_ingredient_pose)[
                    :, : self._dimension_ingredient_distance
                ],
                self._every_other_ingredients,
            )
        )
        reward_distance_between_two_centroids = (
            stir_utils.get_reward_distance_between_two_centroids(
                distance_between_two_centroids,
                self._bowl["radius_bottom"] * 2,
            )
        )

        if reward_distance_between_two_centroids > 0.9:
            return 1000.0, True
        # -------------------------------------------------------

        # -------------------------------------------------------
        (
            mean_array,
            variance_array,
        ) = stir_utils.get_mean_variance_array_delaunay_distance(
            ingredient_positions.reshape(-1, self._length_ingredient_pose)[
                :, :2
            ]
        )
        (
            reward_dyelauna_mean,
            reward_dyelauna_variance,
        ) = stir_utils.get_reward_mean_variance_array_delaunay_distance(
            mean_array,
            variance_array,
            self._bowl["radius_bottom"] * 2,
            self._bowl["radius_bottom"] ** 2,  # TODO(sara): tmp
        )

        if reward_dyelauna_mean > 0.9:
            return 1000.0, True
        if reward_dyelauna_variance > 0.9:
            return 1000.0, True
        # -------------------------------------------------------

        reward = (
            reward_keep_moving_end_effector
            + reward_distance_between_tool_and_ingredient
            + reward_keep_moving_ingredients
            + reward_distance_to_center_pointrd
            + reward_distance_to_center_plane
            + reward_distance_between_two_centroids
            + reward_dyelauna_mean
            + reward_dyelauna_variance
        )

        # if self._check_collision_with_bowl():
        #     return -100, True

        return reward, False

    def step_variables(self, observation: np.ndarray) -> None:
        self._previous_ingredient_positions = observation[
            self._length_tool_pose + self._length_tool_velocity :
        ]
        self._num_step += 1

    def reset_variables(self, observation: np.ndarray) -> None:
        self._every_other_ingredients = False
        self._total_reward_keep_moving_end_effector = 0.0
        self._num_step = 0
        self._previous_ingredient_positions: Optional[np.ndarray] = None
        self._total_reward_array_keep_moving_ingredients = np.zeros(
            self._num_ingredients
        )

    def detect_touch(self, ingredient_positions: np.ndarray) -> bool:
        if (
            self._previous_ingredient_positions is not None
            and (
                np.linalg.norm(
                    ingredient_positions.reshape(
                        -1, self._length_ingredient_pose
                    )[:, : self._dimension_ingredient_distance]
                    - self._previous_ingredient_positions.reshape(
                        -1, self._length_ingredient_pose
                    )[:, : self._dimension_ingredient_distance],
                    axis=1,
                )
                > 0.0000001
            ).any()
        ):
            return True
        else:
            return False
