import numpy as np
import gym
from gym.envs.mujoco import mujoco_env
from typing import Tuple
import os

class LinearActuatorArrayEnv(mujoco_env.MujocoEnv, gym.utils.EzPickle):
  def __init__(self):

    xml_path = 'envs/linear_actuator_array/xmls/linear_actuator_array-v0.xml'
    xml_path = os.path.join(os.getcwd(), xml_path)
    self.frame_skip = 20

    super().__init__(
      xml_path,
      self.frame_skip,
    )

    gym.utils.EzPickle.__init__(self)
  
  def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

    self.sim.step()
    observation = self._get_observation()
    reward, done = self._get_reward(observation)
    info = {}

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
