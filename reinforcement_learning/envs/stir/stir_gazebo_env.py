import random
from enum import Enum
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import tf
from gym.spaces import Box
from stir_ros import Stir

_FRAME_SKIP = 1
_TIME_STEP = 0.1
_ACTION_SIZE = 6
_NUM_INGREDIENTS = 4
_OBSERVATION_SIZE_TOOL = 7
_OBSERVATION_SIZE_INGREDIENTS = 3
_OBSERVATION_SIZE = (
    _OBSERVATION_SIZE_TOOL + _NUM_INGREDIENTS * _OBSERVATION_SIZE_INGREDIENTS
)
_RESET_INGREDIENTS_RADIUS_MIN = 0.02
_RESET_INGREDIENTS_RADIUS_MAX = 0.04
_INGREDIENTS_POSITION_Z = 0.05
_THRESHOLD_DISTANCE = 0.01
_RESET_NOISE_SCALE = 0.0001
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

    def __init__(self, **kwargs) -> None:
        np.random.seed(0)
        random.seed(0)

        self.num_ingredients = _NUM_INGREDIENTS
        self.observation_size_tool = _OBSERVATION_SIZE_TOOL

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(_OBSERVATION_SIZE,),
            dtype=np.float64,
        )

        action_low = np.array(
            [-0.1, -0.1, -0.01, *np.deg2rad([180 - 30, -1.0, -1.0])]
        )
        acthin_high = np.array(
            [0.1, 0.1, 0.01, *np.deg2rad([180 - 20, 1.0, 1.0])]
        )
        self.action_space = Box(
            low=action_low,
            high=acthin_high,
            shape=(_ACTION_SIZE,),
            dtype=np.float32,
        )
        self.stir = Stir()

        self.init_tool_pose = self.stir.get_tool_pose()
        self.num_step = 0

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        q = tf.transformations.quaternion_from_euler(*action[3:])
        pose_target = np.concatenate([action[0:3], q])
        self.stir.step(pose_target)
        observation = self._get_observation()
        reward, terminated = self._get_reward(observation)
        info: Dict[str, str] = {}
        self.num_step += 1

        return observation, reward, terminated, False, info

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

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> np.ndarray:
        tool_position = self.init_tool_pose[0:3] + self.np_random.uniform(
            low=-_RESET_NOISE_SCALE,
            high=_RESET_NOISE_SCALE,
            size=3,
        )
        # tool_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        tool_orientation = self.init_tool_pose[3:]
        self.stir.reset_robot(np.concatenate([tool_position, tool_orientation]))

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

        return observation, {}

    def _get_observation(self) -> np.ndarray:
        return np.concatenate(
            [self.stir.get_tool_pose(), self.stir.get_ingredient_poses()]
        )

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:
        reward = 0.0
        return reward, False
