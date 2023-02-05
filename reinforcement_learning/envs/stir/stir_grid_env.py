import os
import random
from typing import Dict, Tuple

import envs.stir.mujoco_model_utils as mujoco_model_utils
import mujoco
import numpy as np
import yaml
from envs.stir.i_stir_env import IStirEnv
from envs.stir.stir_env_specialization import get_specialization
from gym import utils
from gym.envs.mujoco import MujocoEnv

_FRAME_SKIP = 40
_TIME_STEP = 0.0025
_TOOL_POSE_DIMENSION = 7
_INGREDIENT_POSE_DIMENSION = 7
_TOOL_POSE_INDEX = 0
_INGREDIENTS_POSE_INDEX = _TOOL_POSE_DIMENSION
_RESET_INGREDIENTS_RADIUS_MIN = 0.01
_RESET_INGREDIENTS_RADIUS_MAX = 0.03
_THRESHOLD_RESET_REWARD = 0.9
_RESET_NOISE_SCALE = 0.01
_MAX_TRIAL_INGREDIENT_RANDOMIZATION = 100


class StirGridEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": np.round(1.0 / (_TIME_STEP * _FRAME_SKIP)),
    }

    def __init__(self, specialization, xml, **kwargs) -> None:

        np.random.seed(1)
        random.seed(1)

        # create model
        env_directory = os.path.abspath(os.path.dirname(__file__))
        xml_file_path: str = os.path.join(
            env_directory,
            "xmls",
            f"{xml}.xml",
        )
        mesh_dir_path: str = os.path.join(env_directory, "meshes")

        assets = dict()
        if os.path.exists(mesh_dir_path):
            mesh_files = os.listdir(mesh_dir_path)
            for file in mesh_files:
                with open(os.path.join(mesh_dir_path, file), "rb") as f:
                    assets[file] = f.read()

        self.model = mujoco.MjModel.from_xml_path(xml_file_path, assets)
        self.data = mujoco.MjData(self.model)

        # get init_tool_pose
        self._init_tool_pose = self.data.qpos[
            _TOOL_POSE_INDEX : _TOOL_POSE_INDEX + _TOOL_POSE_DIMENSION
        ]
        if self._init_tool_pose.shape == (7,):  # wxyz -> xyzw
            self._init_tool_pose = np.append(
                np.delete(self._init_tool_pose, 3), self._init_tool_pose[3]
            )

        # get config param
        env_directory = os.path.abspath(os.path.dirname(__file__))
        tool_param_path = os.path.join(
            env_directory, "config", "tool_param.yaml"
        )
        bowl_param_path = os.path.join(
            env_directory, "config", "bowl_param.yaml"
        )
        with open(tool_param_path, "r") as f:
            tool = yaml.safe_load(f)
        with open(bowl_param_path, "r") as f:
            bowl = yaml.safe_load(f)
            for key in bowl:
                bowl[key] -= tool["radius_smallest_circle"]

        self._num_ingredients = mujoco_model_utils.get_num_ingredient(
            self.model
        )
        self._get_tool_geom_ids = mujoco_model_utils.get_tool_geom_ids(
            self.model, tool["num_geom"]
        )
        self._bowl_geom_ids = mujoco_model_utils.get_bowl_geom_ids(
            self.model, int(bowl["num_geom"])
        )

        self._stir_env: IStirEnv = get_specialization(
            specialization,
            self._init_tool_pose,
            self._num_ingredients,
            self.check_collision_with_bowl,
        )

        utils.EzPickle.__init__(**locals())

        # MujocoEnv overrides action_space
        MujocoEnv.__init__(
            self,
            xml_file_path,
            _FRAME_SKIP,
            observation_space=self._stir_env.observation_space,
            **kwargs,
        )

        if self._stir_env.action_space is not None:
            self.action_space = self._stir_env.action_space

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        control = self._stir_env.get_controller_input(action)
        self.data.ctrl[:] = control

        # for _ in range(self.frame_skip):
        #     self.data.qpos[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + _TOOL_POSE_DIMENSION] *= (
        #         self._stir_env.observation_tool_pose
        #         + np.logical_not(self._stir_env.observation_tool_pose)
        #         * self._init_tool_pose
        #     )
        #     mujoco.mj_step(self.model, self.data, nstep=1)

        # self.data.qpos[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + _TOOL_POSE_DIMENSION] *= (
        #     self._stir_env.observation_tool_pose
        #     + np.logical_not(self._stir_env.observation_tool_pose) * self._init_tool_pose
        # )

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        observation = self._get_observation()
        reward, terminated = self._stir_env.get_reward(observation)
        self._stir_env.step_variables(observation)
        info: Dict[str, str] = {}

        if self.render_mode == "human":
            self.render()

        # print(self.num_step / 40)

        return observation, reward, terminated, False, info

    def reset_model(self) -> np.ndarray:

        if self._num_ingredients > 0:
            for _ in range(_MAX_TRIAL_INGREDIENT_RANDOMIZATION):
                mujoco.mj_resetData(self.model, self.data)
                for i in range(self._num_ingredients):
                    radius = np.random.uniform(
                        _RESET_INGREDIENTS_RADIUS_MIN,
                        _RESET_INGREDIENTS_RADIUS_MAX,
                    )
                    angle = np.random.uniform(0.0, 2 * np.pi)
                    ingredient_pose_index = (
                        _INGREDIENTS_POSE_INDEX + i * _INGREDIENT_POSE_DIMENSION
                    )
                    self.data.qpos[ingredient_pose_index] = radius * np.cos(
                        angle
                    )
                    self.data.qpos[ingredient_pose_index + 1] = radius * np.sin(
                        angle
                    )
                mujoco.mj_forward(self.model, self.data)
                reward, terminated = self._stir_env.get_reward(
                    self._get_observation()
                )
                if not terminated and reward < _THRESHOLD_RESET_REWARD:
                    break

        else:
            mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[
            _TOOL_POSE_INDEX : _TOOL_POSE_INDEX + _TOOL_POSE_DIMENSION
        ][self._stir_env.observation_tool_pose] += self.np_random.uniform(
            low=-_RESET_NOISE_SCALE,
            high=_RESET_NOISE_SCALE,
            size=sum(self._stir_env.observation_tool_pose),
        )
        self.data.qvel[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + 6][
            self._stir_env.observation_tool_velocity
        ] += self.np_random.uniform(
            low=-_RESET_NOISE_SCALE,
            high=_RESET_NOISE_SCALE,
            size=sum(self._stir_env.observation_tool_velocity),
        )

        mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        self._stir_env.reset_variables(observation)

        return observation

    def _get_observation(self) -> np.ndarray:
        tool_pose = self.data.qpos[
            _TOOL_POSE_INDEX : _TOOL_POSE_INDEX + _TOOL_POSE_DIMENSION
        ][self._stir_env.observation_tool_pose]
        # wxyz -> xyzw
        if tool_pose.shape == (7,):
            tool_pose = np.append(np.delete(tool_pose, 3), tool_pose[3])
        tool_velocity = self.data.qvel[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + 6][
            self._stir_env.observation_tool_velocity
        ]

        # ingredient_pose: wxyz
        ingredient_pose = (
            self.data.qpos[_INGREDIENTS_POSE_INDEX:]
            .reshape(-1, _INGREDIENT_POSE_DIMENSION)[
                :, self._stir_env.observation_ingredient_pose
            ]
            .flatten()
        )

        observation = np.concatenate(
            [tool_pose, tool_velocity, ingredient_pose]
        )
        return observation

    def check_collision_with_bowl(self) -> bool:
        for i in range(self.data.contact.geom1.size):
            if self.data.contact.geom1[i] in self._bowl_geom_ids:
                if self.data.contact.geom2[i] in self._tool_geom_ids:
                    return True
            if self.data.contact.geom2[i] in self._tool_geom_ids:
                if self.data.contact.geom2[i] in self._bowl_geom_ids:
                    return True
        return False
