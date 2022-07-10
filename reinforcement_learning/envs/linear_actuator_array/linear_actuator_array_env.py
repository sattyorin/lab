import os
from typing import Dict, Tuple

import mujoco
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

        model = mujoco.MjModel.from_xml_path(xml_file)
        self.num_module = model.names.decode("UTF-8").count("module")
        self.num_object = model.names.decode("UTF-8").count("object")
        self.module_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"module{i}")
                for i in range(self.num_module)
            ]
        )
        self.object_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object{i}")
                for i in range(self.num_object)
            ]
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

        modules = np.array(self.data.xpos[self.module_ids, 2])  # 2: z
        objects = np.array(self.data.xpos[self.object_ids, 2])

        return np.concatenate(
            [modules, objects, np.array([self.dt])], dtype=float
        )

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:

        reward = 0.0
        done = False

        return reward, done
