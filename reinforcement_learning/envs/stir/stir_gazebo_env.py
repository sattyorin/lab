import random
from typing import Dict, Tuple

import gym
import numpy as np
from gazebo_bridge.bridge import GazeboBridge
from gym.spaces import Box

_FRAME_SKIP = 1
_TIME_STEP = 0.1
_ENV = "stir-v0"
_OBSERVATION_SIZE = 1
_ACTION_SIZE = 7


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

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(_OBSERVATION_SIZE,),
            dtype=np.float64,
        )

        self.action_space = Box(
            low=-np.inf, high=np.inf, shape=(_ACTION_SIZE,), dtype=np.float32
        )
        self.bridge = GazeboBridge()

    def _do_simulation(self, action: np.ndarray) -> None:
        self.bridge.step(action)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._do_simulation(action)
        observation = self._get_observation()
        reward, terminated = self._get_reward()
        info: Dict[str, str] = {}

        return observation, reward, terminated, False, info

    def reset(self) -> np.ndarray:
        self.bridge.reset()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        observation = np.zeros(self.observation_space.shape[0])
        return observation

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:
        reward = 0.0
        return reward, False
