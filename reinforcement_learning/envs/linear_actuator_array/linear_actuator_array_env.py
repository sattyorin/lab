import os
from typing import Dict, Tuple

import mujoco
import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
from scipy import interpolate

import envs.linear_actuator_array.linear_actuator_array_config as config

_FRAME_SKIP = 20


class LinearActuatorArrayEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": np.round(1.0 / (config.timestep * _FRAME_SKIP)),
    }

    def __init__(self, **kwargs) -> None:

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

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_module + self.num_object + 1,),
            dtype=np.float64,
        )

        utils.EzPickle.__init__(**locals())

        MujocoEnv.__init__(
            self,
            xml_file,
            _FRAME_SKIP,
            observation_space=observation_space,
            **kwargs,
        )

        self.flow_time_interval = config.flow_time_interval
        self.num_valid_object = self.num_object

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:

        self.do_simulation(action, self.frame_skip)
        observation = self._get_observation()
        reward, terminated = self._get_reward(observation)
        info: Dict[str, str] = {}

        self.renderer.render_step()
        return observation, reward, terminated, False, info

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

        self.num_valid_object = self.num_object

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:

        modules = np.array(self.data.xpos[self.module_ids, 2])  # 2: z
        objects = np.array(self.data.xpos[self.object_ids, 2])

        return np.concatenate(
            [modules, objects, np.array([self.data.time])], dtype=float
        )

    def _get_reward(self, observation: np.ndarray) -> Tuple[float, bool]:

        terminated = False
        reward = 0.0

        previous_num_valid_object = self.num_valid_object

        self.num_valid_object = sum(
            config.palm_height
            > observation[self.num_module : self.num_module + self.num_object]
        )

        if self.num_valid_object:
            terminated = True

        if previous_num_valid_object != self.num_valid_object:
            reward = abs(
                observation[-1]
                - self.flow_time_interval
                * (self.num_object - self.num_valid_object)
            )

        return reward, terminated

    def update_reward(self, reward: np.ndarray) -> np.ndarray:
        k = 2
        if self.num_object > 3:
            k = 3
        x = np.where(reward > 0)[0]
        y = reward[x]
        t, c, k = interpolate.splrep(x, y, s=0, k=k)

        N = len(reward)
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, N)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        return spline(xx)
