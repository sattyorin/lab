import os
from typing import Dict, Tuple

import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv

_FRAME_SKIP = 20


class LinearActuatorArrayEnv(MujocoEnv, utils.EzPickle):
    def __init__(self) -> None:

        xml_file: str = os.path.join(
            os.getcwd(),
            "envs/linear_actuator_array/xmls/linear_actuator_array-v0.xml",
        )

        utils.EzPickle.__init__(self)

        MujocoEnv.__init__(self, xml_file, _FRAME_SKIP)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        self.do_simulation(action, self.frame_skip)
        observation = self._get_observation()
        reward, done = self._get_reward(observation)
        info: Dict[str, str] = {}

        return observation, reward, done, info

    def reset_model(self) -> np.ndarray:

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:

        observation = np.array([])
        return observation

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:

        reward = 0.0
        done = False

        return reward, done
