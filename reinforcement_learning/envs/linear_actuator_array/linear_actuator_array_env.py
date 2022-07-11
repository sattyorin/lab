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

        import mujoco_py  # TODO(sara): use only mujoco

        py_model = mujoco_py.load_model_from_path(xml_file)

        self.num_module = sum(n.count("module") for n in py_model.body_names)
        self.num_object = sum(n.count("object") for n in py_model.body_names)

        self.module_ids = np.array(
            [
                py_model.body_name2id(f"module{i}")
                for i in range(self.num_module)
            ]
        )
        self.object_ids = np.array(
            [
                py_model.body_name2id(f"object{i}")
                for i in range(self.num_object)
            ]
        )

        """ use mujoco
        model = mujoco.MjModel.from_xml_path(xml_file)
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
        """

        self.range_x, self.range_y = (
            py_model.body_pos[self.module_ids[-1], 0:2]
            + py_model.geom_size[py_model.geom_name2id("object0_geom"), 0:2]
        )

        """ use mujoco
        self.range_x, self.range_y = (
            model.body_pos[self.module_ids[-1], 0:2]
            + model.geom_size[
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, "object0_geom"
                ),
                0:2,
            ]
        )
        """

        self.object_qpos_addrs = [
            py_model.get_joint_qpos_addr(f"object{i}_joint")[0]
            for i in range(self.num_object)
        ]

        utils.EzPickle.__init__(self)

        MujocoEnv.__init__(self, xml_file, _FRAME_SKIP)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        self.do_simulation(action, self.frame_skip)
        observation = self._get_observation()
        reward, done = self._get_reward(observation)
        info: Dict[str, str] = {}

        return observation, reward, done, info

    def reset_model(self) -> np.ndarray:

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        for object_addr in self.object_qpos_addrs:
            self.data.qpos[object_addr] = np.random.uniform(
                -self.range_x, self.range_x, size=self.num_object
            )
            self.data.qpos[object_addr + 1] = np.random.uniform(
                -self.range_y, self.range_y, size=self.num_object
            )
        mujoco.mj_forward(self.model, self.data)

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
