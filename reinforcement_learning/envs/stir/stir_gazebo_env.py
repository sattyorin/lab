import os
import random
from enum import Enum
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import tf
import yaml
from envs.stir.i_stir_env import IStirEnv
from envs.stir.stir_env_specialization import get_specialization
from stir_ros import Stir

_FRAME_SKIP = 1
_TIME_STEP = 0.1
_RESET_INGREDIENTS_RADIUS_MIN = 0.02
_RESET_INGREDIENTS_RADIUS_MAX = 0.04
_THRESHOLD_RESET_REWARD = 0.9
_MAX_TRIAL_INGREDIENT_RANDOMIZATION = 100


class Observation(Enum):
    TOOL_POSITION_X = 0
    TOOL_POSITION_Y = 1
    TOOL_POSITION_Z = 2
    TOOL_ORIENTATION_X = 3
    TOOL_ORIENTATION_Y = 4
    TOOL_ORIENTATION_Z = 5
    TOOL_ORIENTATION_W = 6


class StirGazeboEnv(gym.Env):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": np.round(1.0 / (_TIME_STEP * _FRAME_SKIP)),
    }

    def __init__(self, specialization, **kwargs) -> None:
        np.random.seed(0)
        random.seed(0)

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

        self._init_tool_pose = np.array(
            [0.323, 0.0, -0.087, 0.0, 0.961, 0.0, 0.277]
        )  # TODO(sara): get pose from somewhere

        self.stir = Stir(self._init_tool_pose)
        self._num_ingredients = self.stir.num_ingredients

        self._stir_env: IStirEnv = get_specialization(
            specialization,
            self._init_tool_pose,
            self._num_ingredients,
            self.check_collision_with_bowl,
        )

        self.observation_space = self._stir_env.observation_space
        self.action_space = self._stir_env.action_space

        if self._stir_env.is_position_controller:
            print("----------")
            print("stir_gazebo_env: use position controller")
            print("----------")
            self._step = self.step_position_controller
        elif self._stir_env.is_velocity_controller:
            print("----------")
            print("stir_gazebo_env: use velocity controller")
            print("----------")
            self._step = self.step_velocity_controller
        else:
            raise ValueError("controller not selected")

        self._previous_sec: Optional[float] = None

    def step_position_controller(self, action: np.ndarray) -> None:
        q = tf.transformations.quaternion_from_euler(*action[3:])
        pose_target = np.concatenate([action[0:3], q])
        self.stir.step_position_controller(
            pose_target, self._stir_env.observation_tool_pose
        )

    def step_velocity_controller(self, action: np.ndarray) -> None:
        self.stir.step_velocity_controller(
            action, self._stir_env.observation_tool_pose
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step(self._stir_env.get_controller_input(action))
        observation = self._get_observation()
        reward, terminated = self._stir_env.get_reward(observation)
        self._stir_env.step_variables(observation)
        info: Dict[str, str] = {}

        return observation, reward, terminated, False, info

    def render(self):
        return None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> np.ndarray:

        self.stir.reset_robot()

        if self._num_ingredients > 0:
            for _ in range(_MAX_TRIAL_INGREDIENT_RANDOMIZATION):
                radius = np.random.uniform(
                    _RESET_INGREDIENTS_RADIUS_MIN,
                    _RESET_INGREDIENTS_RADIUS_MAX,
                    size=self._num_ingredients,
                )
                angle = np.random.uniform(
                    0.0, 2 * np.pi, size=self._num_ingredients
                )
                init_ingredient_poses = np.vstack(
                    [
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        (self._init_tool_pose[2])
                        * np.ones(self._num_ingredients),
                    ]
                ).transpose()

                init_ingredient_poses[:, 0] += self._init_tool_pose[0]
                init_ingredient_poses[:, 1] += self._init_tool_pose[1]

                # TODO(sara): self._stir_env.observationingredient_pose[:3]?
                reward, terminated = self._stir_env.get_reward(
                    self._get_observation()
                )
                if not terminated and reward < _THRESHOLD_RESET_REWARD:
                    break

        observation = self._get_observation()
        self._stir_env.reset_variables(observation)
        self._previous_sec: Optional[float] = None
        return observation, {}

    def _get_observation(self) -> np.ndarray:
        tool_pose, sec = self.stir.get_tool_pose()
        tool_pose_euler = np.concatenate(
            [
                tool_pose[:3],
                np.array(
                    tf.transformations.euler_from_quaternion(tool_pose[3:])
                ),
            ]
        )
        if (
            not self._previous_sec
            or abs(sec - self._previous_sec) < np.finfo(float).eps
        ):
            tool_velocity = np.zeros(6, dtype=float)
        else:
            tool_velocity = (
                tool_pose_euler - self._previous_tool_pose_euler
            ) / (sec - self._previous_sec)

        observation = np.concatenate(
            [
                tool_pose[self._stir_env.observation_tool_pose],
                tool_velocity[self._stir_env.observation_tool_velocity],
                self.stir.get_ingredient_poses()
                .reshape(-1, 7)[:, self._stir_env.observation_ingredient_pose]
                .flatten(),
            ]
        )
        self._previous_tool_pose_euler = tool_pose_euler
        self._previous_sec = sec
        return observation

    def check_collision_with_bowl(
        self, end_pose: np.ndarray, base_pose: np.ndarray
    ) -> bool:
        # TODO(sara): do the test
        bowl_top_position_z = (
            self._init_tool_pose[2]
            + self._bowl["bowl_height"]
            - self._bowl["bowl_bottom_to_init_z"]
        )
        bowl_bottom_position_z = (
            self._init_tool_pose[2] - self._bowl["bowl_bottom_to_init_z"]
        )

        # tool end condition
        a = (
            self._bowl["bowl_radius_top"] - self._bowl["bowl_radius_bottom"]
        ) / (bowl_top_position_z - bowl_bottom_position_z)
        if np.hypot(*(end_pose[:2] - self._init_tool_pose[:2])) > self._bowl[
            "bowl_radius_bottom"
        ] + a * (end_pose[2] - bowl_bottom_position_z):
            return True

        # piercing condition
        a_xy = (base_pose[1] - end_pose[1]) / (base_pose[0] - end_pose[0])
        b_xy = -a_xy * end_pose[0] + end_pose[1]

        a = 1 + a_xy
        b = a_xy * b_xy
        c = b_xy**2 - self._bowl["bowl_radius_top"] ** 2
        x1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        x2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        y1 = a_xy * x1 + b_xy
        y2 = a_xy * x2 + b_xy

        if 0 < np.dot(
            base_pose[:2] - end_pose[:2],
            np.array([x1, y1]) - end_pose[:2],
        ):
            x = x1
            y = y1
        else:
            x = x2
            y = y2

        distance_end_to_bowl_top_xy = np.hypot(
            *(np.array([x, y]) - end_pose[:2])
        )

        distance_xy = np.hypot(*(end_pose[:2] - base_pose[:2]))
        a_wz1 = (base_pose[2] - end_pose[2]) / distance_xy
        b_wz1 = end_pose[2]
        return bowl_top_position_z > a_wz1 * distance_end_to_bowl_top_xy + b_wz1
