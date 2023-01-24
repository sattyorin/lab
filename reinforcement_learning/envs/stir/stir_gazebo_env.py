import random
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import tf
from gym.spaces import Box
from stir_ros import Stir

_ARM_CONTROLLER = "position"
_FRAME_SKIP = 1
_TIME_STEP = 0.1
_ACTION_SIZE = 6
_NUM_INGREDIENTS = 0  # TODO(sara)
_OBSERVATION_SIZE_TOOL = 7 + 6
_OBSERVATION_SIZE_INGREDIENTS = 3
_OBSERVATION_SIZE = (
    _OBSERVATION_SIZE_TOOL + _NUM_INGREDIENTS * _OBSERVATION_SIZE_INGREDIENTS
)
_RESET_INGREDIENTS_RADIUS_MIN = 0.02
_RESET_INGREDIENTS_RADIUS_MAX = 0.04
_INGREDIENTS_POSITION_Z = 0.05
_THRESHOLD_DISTANCE = 0.01
_MAX_TRIAL_INGREDIENT_RANDOMIZATION = 100
_TARGET_VELOCITY = 0.02
# _BOWL_RADIUS_TOP = 0.12
# _BOWL_RADIUS_BOTTOM = 0.08
# _BOWL_TOP_POSITION_Z = -0.139 + 0.06 - 0.04  # init_tool_pose + height + alpha
_BOWL_RADIUS_TOP = 0.11
_BOWL_RADIUS_BOTTOM = 0.07
_BOWL_TOP_POSITION_Z = -0.129 + 0.05 - 0.03  # init_tool_pose + height + alpha
_BOWL_BOTTOM_POSITION_Z = -0.139


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

    def __init__(self, **kwargs) -> None:
        np.random.seed(0)
        random.seed(0)

        self.num_ingredients = _NUM_INGREDIENTS
        self.observation_size_tool = _OBSERVATION_SIZE_TOOL
        self.num_step = 0
        self._previous_tool_pose_euler = np.zeros(6, dtype=float)
        self._previous_sec: Optional[float] = None

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(_OBSERVATION_SIZE,),
            dtype=np.float64,
        )

        self.init_tool_pose = np.array(
            [0.374, -0.015, -0.139, -0.184, 0.936, -0.279, 0.114]
        )  # TODO(sara): get pose from somewhere

        self.stir = Stir(self.init_tool_pose)

        euler = tf.transformations.euler_from_quaternion(
            self.init_tool_pose[3:]
        )

        if _ARM_CONTROLLER == "position":
            self._step = self.step_position_controller
            init_tool_euler_pose = np.concatenate(
                [self.init_tool_pose[:3], euler]
            )
            action_low = (
                np.array([-0.05, -0.05, -0.05, -0.1, -0.1, -0.1])
                + init_tool_euler_pose
            )
            action_high = (
                np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
                + init_tool_euler_pose
            )
        elif _ARM_CONTROLLER == "velocity":
            self._step = self.step_position_controller
            action_low = -0.1
            action_high = 0.1
        else:
            raise ValueError("_ARM_CONTROLLER is invalid")

        self.action_space = Box(
            low=action_low,
            high=action_high,
            shape=(_ACTION_SIZE,),
            dtype=np.float32,
        )

    def step_position_controller(self, action: np.ndarray) -> None:
        q = tf.transformations.quaternion_from_euler(*action[3:])
        pose_target = np.concatenate([action[0:3], q])
        self.stir.step_position_controller(pose_target)

    def step_velocity_controller(self, action: np.ndarray) -> None:
        self.stir.step_velocity_controller(action)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step(action)
        observation = self._get_observation()
        reward, terminated = self._get_reward(observation)
        info: Dict[str, str] = {}
        self.num_step += 1

        return observation, reward, terminated, False, info

    def render(self):
        return None

    def _get_distance_between_two_centroids(
        self, observation: np.ndarray
    ) -> float:
        ingredients = observation[self.observation_size_tool :].reshape(-1, 3)
        centroid1 = np.mean(
            ingredients[0 : (self.num_ingredients // 2)], axis=0
        )
        centroid2 = np.mean(ingredients[(self.num_ingredients // 2) :], axis=0)
        distance = np.linalg.norm(centroid1[0:2] - centroid2[0:2])

        return distance

    def get_small_velocity_reward(self, velocity: float) -> float:
        return 1 - np.exp(
            -velocity / (_TARGET_VELOCITY - _TARGET_VELOCITY * 0.7)
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> np.ndarray:

        self.stir.reset_robot()

        if self.num_ingredients > 0:
            for _ in range(_MAX_TRIAL_INGREDIENT_RANDOMIZATION):
                radius = np.random.uniform(
                    _RESET_INGREDIENTS_RADIUS_MIN,
                    _RESET_INGREDIENTS_RADIUS_MAX,
                    size=self.num_ingredients,
                )
                angle = np.random.uniform(
                    0.0, 2 * np.pi, size=self.num_ingredients
                )
                init_ingredient_poses = np.vstack(
                    [
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        _INGREDIENTS_POSITION_Z * np.ones(self.num_ingredients),
                    ]
                ).transpose()

                self.stir.reset_ingredient(init_ingredient_poses)
                observation = self._get_observation()
                if (
                    self._get_distance_between_two_centroids(observation)
                    > _THRESHOLD_DISTANCE * 2.0
                ):
                    break
        else:
            observation = self._get_observation()

        self.num_step = 0
        self._previous_sec = None

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
            velocity = np.zeros(6, dtype=float)
        else:
            velocity = (tool_pose_euler - self._previous_tool_pose_euler) / (
                sec - self._previous_sec
            )

        observation = np.concatenate(
            [
                tool_pose,
                velocity,
                # self.stir.get_ingredient_poses(),
            ]
        )
        self._previous_tool_pose_euler = tool_pose_euler
        self._previous_sec = sec
        return observation

    def _check_collision_with_bowl(
        self, end_pose: np.ndarray, base_pose: np.ndarray
    ) -> bool:

        # tool end condition
        a = (_BOWL_RADIUS_TOP - _BOWL_RADIUS_BOTTOM) / (
            _BOWL_TOP_POSITION_Z - _BOWL_BOTTOM_POSITION_Z
        )
        if np.hypot(
            *(end_pose[:2] - self.init_tool_pose[:2])
        ) > _BOWL_RADIUS_BOTTOM + a * (end_pose[2] - _BOWL_BOTTOM_POSITION_Z):
            return True

        # piercing condition
        a_xy = (base_pose[1] - end_pose[1]) / (base_pose[0] - end_pose[0])
        b_xy = -a_xy * end_pose[0] + end_pose[1]

        a = 1 + a_xy
        b = a_xy * b_xy
        c = b_xy**2 - _BOWL_RADIUS_TOP**2
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
        return (
            _BOWL_TOP_POSITION_Z > a_wz1 * distance_end_to_bowl_top_xy + b_wz1
        )

    def get_distance_score(self, distance: float) -> float:
        return (1 + np.tanh((-abs(distance) * 200 + 5) / 2)) / 2

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:
        reward = 0.0
        tool_pose = observation[:7]
        # tool_velocity = observation[:7]

        # velocity = np.linalg.norm(np.array([observation[7], observation[8]]))
        # reward_small_velocity = self.get_small_velocity_reward(velocity)

        reward = self.get_distance_score(tool_pose[2] - self.init_tool_pose[2])

        if self.init_tool_pose[2] - tool_pose[2] > 0.028:
            return reward, True

        if tool_pose[2] - self.init_tool_pose[2] > 0.02:
            return reward, True

        if self._check_collision_with_bowl(
            self.stir.get_tool_pose()[0], self.stir.get_gripper_pose()[0]
        ):
            return reward, True

        return reward, False
