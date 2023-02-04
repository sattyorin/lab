from typing import Optional, Tuple

import numpy as np
from gym.spaces import Box


class IStirEnv:
    def __init__(
        self,
        is_position_controller: bool,
        is_velocity_controller: bool,
        action_is_position: bool,
        action_is_velocity: bool,
        observation_tool_pose: np.ndarray,
        observation_tool_velocity: np.ndarray,
        observation_ingredient_pose: np.ndarray,
        action_space: Optional[Box],
        num_ingredients: int,
    ) -> None:
        """
        observation_tool_pose: [position.x, position.y, position.z,
                                quaternion.w, quaternion.x, quaternion.y, quaternion.z]
        observation_tool_velocity: [x, y, z, roll, pitch, yaw]
        observation_ingredient_position: [x, y, z]
        """
        if observation_tool_pose.shape != (7,):
            raise ValueError(
                f"observation_tool_pose shape should be (7,), but is {observation_tool_pose.shape}"
            )

        if observation_tool_velocity.shape != (6,):
            raise ValueError(
                f"observation_tool_velocity shape should be (7,), but is {observation_tool_velocity.shape}"
            )

        if observation_ingredient_pose.shape != (7,):
            raise ValueError(
                f"observation_ingredient_position shape should be (7,), but is {observation_ingredient_pose.shape}"
            )

        self.is_position_controller = is_position_controller
        self.is_velocity_controller = is_velocity_controller
        self.observation_tool_pose = observation_tool_pose
        self.observation_tool_velocity = observation_tool_velocity
        self.observation_ingredient_pose = observation_ingredient_pose
        if (
            action_space is not None
        ):  # MujocoEnv generates action_space automatically
            self.action_space = action_space

        self.observation_space: Box = Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                sum(observation_tool_pose)
                + sum(observation_tool_velocity)
                + sum(observation_ingredient_pose) * num_ingredients,
            ),
            dtype=np.float64,
        )

        self._length_tool_pose = sum(self.observation_tool_pose)
        self._length_tool_velocity = sum(self.observation_tool_velocity)
        self._length_ingredient_pose = sum(self.observation_ingredient_pose)

        self._dimension_ingredient_distance: Optional[int] = None
        if not (
            self.observation_ingredient_pose[:3] ^ np.array([True, True, False])
        ).all():
            self._dimension_ingredient_distance = 2
        elif not (
            self.observation_ingredient_pose[:3] ^ np.array([True, True, True])
        ).all():
            self._dimension_ingredient_distance = 3

    def get_controller_input(self, action: np.ndarray) -> np.ndarray:
        """
        Args:
            action:
        Returns:
            np.ndarray: input of controller
        """
        raise NotImplementedError

    def get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:
        """
        Args:
            observation:
        Returns:
            bool: reward
            int: terminated
        """
        raise NotImplementedError

    def step_variables(self, observation: np.ndarray) -> None:
        raise NotImplementedError

    def reset_variables(self, observation: np.ndarray) -> None:
        raise NotImplementedError
