import os
import random
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np

# TODO(sara): add switcher
from envs.stir.stir_env_xyz_move_ingredients import (
    StirEnvXYZMoveIngredients as StirEnv,
)
from envs.stir.stir_util import get_distance_between_two_centroids
from gym import utils
from gym.envs.mujoco import MujocoEnv

_FRAME_SKIP = 40
_TIME_STEP = 0.0025
_ENV = "stir-v2"  # xml
_TOOL_POSE_INDEX = 0
_INGREDIENTS_POSE_INDEX = 7
_RESET_INGREDIENTS_RADIUS_MIN = 0.01
_RESET_INGREDIENTS_RADIUS_MAX = 0.03
_TOOL_GEOM_NUM = 2  # TODO(sara): get num from model
_BOWL_GEOM_NUM = 16  # TODO(sara): get num from model
_THRESHOLD_RESET_DISTANCE = 0.02
_RESET_NOISE_SCALE = 0.01
_MAX_TRIAL_INGREDIENT_RANDOMIZATION = 100


class StirMujocoEnv(MujocoEnv, utils.EzPickle, StirEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": np.round(1.0 / (_TIME_STEP * _FRAME_SKIP)),
    }

    def __init__(self, **kwargs) -> None:

        np.random.seed(1)
        random.seed(1)

        # create model
        env_directory = os.path.abspath(os.path.dirname(__file__))
        xml_file_path: str = os.path.join(
            env_directory,
            "xmls",
            f"{_ENV}.xml",
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
        self._init_tool_pose = self.data.qpos[
            _TOOL_POSE_INDEX : _TOOL_POSE_INDEX + 7
        ]
        self._init_tool_pose = np.append(
            np.delete(self._init_tool_pose, 3), self._init_tool_pose[3]
        )

        # get num_ingredients
        self._num_ingredients = 0
        for first_adr in self.model.name_bodyadr:
            adr = first_adr
            while self.model.names[adr] != 0:
                adr += 1
            if self.model.names[first_adr:adr].decode().count("ingredient") > 0:
                self._num_ingredients += 1

        self._get_ids()

        StirEnv.__init__(self, self._init_tool_pose)

        utils.EzPickle.__init__(**locals())

        # MujocoEnv overrides action_space
        action_space = self.action_space
        MujocoEnv.__init__(
            self,
            xml_file_path,
            _FRAME_SKIP,
            observation_space=self.observation_space,
            **kwargs,
        )
        self.action_space = action_space

        self._total_velocity_reward = 0.0
        self.num_step = 0
        self.previous_angle = 0.0  # TODO(sara): generalize it
        self._every_other_ingredients = False
        self._previous_ingredient_positions: Optional[np.ndarray] = None

    def _get_ids(self):
        # self._ingredient_ids = np.array(
        #     [
        #         mujoco.mj_name2id(
        #             self.model, mujoco.mjtObj.mjOBJ_BODY, f"ingredient{i}"
        #         )
        #         for i in range(self._num_ingredients)
        #     ]
        # )
        # self._tool_id = mujoco.mj_name2id(
        #     self.model, mujoco.mjtObj.mjOBJ_BODY, "tools"
        # )
        # self._ingredient_geom_ids = np.array(
        #     [
        #         mujoco.mj_name2id(
        #             self.model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_ingredient{i}"
        #         )
        #         for i in range(self._num_ingredients + 1)
        #     ]
        # )
        self._tool_geom_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_tool{i}"
                )
                for i in range(_TOOL_GEOM_NUM + 1)
            ]
        )

        self._bowl_geom_ids = np.insert(
            np.array(
                [
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        f"geom_bowl_fragment{i}",
                    )
                    for i in range(1, _BOWL_GEOM_NUM + 1)
                ]
            ),
            0,
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "geom_bowl_bottom0",
            ),
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        control = self._get_controller_input(action)
        self.data.ctrl[:] = control

        # for _ in range(self.frame_skip):
        #     self.data.qpos[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + 7] *= (
        #         self._observation_tool_pose
        #         + np.logical_not(self._observation_tool_pose)
        #         * self._init_tool_pose
        #     )
        #     mujoco.mj_step(self.model, self.data, nstep=1)

        # self.data.qpos[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + 7] *= (
        #     self._observation_tool_pose
        #     + np.logical_not(self._observation_tool_pose) * self._init_tool_pose
        # )
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        observation = self._get_observation()
        reward, terminated = self._get_reward(observation)
        info: Dict[str, str] = {}

        if self.render_mode == "human":
            self.render()

        self.num_step += 1
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
                    ingredient_pose_index = _INGREDIENTS_POSE_INDEX + i * 7
                    self.data.qpos[ingredient_pose_index] = radius * np.cos(
                        angle
                    )
                    self.data.qpos[ingredient_pose_index + 1] = radius * np.sin(
                        angle
                    )
                mujoco.mj_forward(self.model, self.data)
                if (
                    get_distance_between_two_centroids(
                        self.data.qpos[_INGREDIENTS_POSE_INDEX:].reshape(-1, 7)[
                            :, :2
                        ],
                        self._every_other_ingredients,
                    )
                    > _THRESHOLD_RESET_DISTANCE
                ):
                    break

        else:
            mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + 7][
            self._observation_tool_pose
        ] += self.np_random.uniform(
            low=-_RESET_NOISE_SCALE,
            high=_RESET_NOISE_SCALE,
            size=sum(self._observation_tool_pose),
        )
        self.data.qvel[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + 6][
            self._observation_tool_velocity
        ] += self.np_random.uniform(
            low=-_RESET_NOISE_SCALE,
            high=_RESET_NOISE_SCALE,
            size=sum(self._observation_tool_velocity),
        )

        mujoco.mj_forward(self.model, self.data)

        self._total_velocity_reward = 0.0
        self.num_step = 0
        self._previous_angle = 0.0  # TODO(sara): generalize it
        self._every_other_ingredients = False
        self._previous_ingredient_positions: Optional[np.ndarray] = None

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        tool_pose = self.data.qpos[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + 7][
            self._observation_tool_pose
        ]
        # wxyz -> xyzw
        if tool_pose.shape == (7,):
            tool_pose = np.append(np.delete(tool_pose, 3), tool_pose[3])
        tool_velocity = self.data.qvel[_TOOL_POSE_INDEX : _TOOL_POSE_INDEX + 6][
            self._observation_tool_velocity
        ]

        # ingredient_pose: wxyz
        ingredient_pose = (
            self.data.qpos[_INGREDIENTS_POSE_INDEX:]
            .reshape(-1, 7)[:, self._observation_ingredient_pose]
            .flatten()
        )

        observation = np.concatenate(
            [tool_pose, tool_velocity, ingredient_pose]
        )
        return observation

    def _check_collision_with_bowl(self) -> bool:
        for i in range(self.data.contact.geom1.size):
            if self.data.contact.geom1[i] in self._bowl_geom_ids:
                if self.data.contact.geom2[i] in self._tool_geom_ids:
                    return True
            if self.data.contact.geom2[i] in self._tool_geom_ids:
                if self.data.contact.geom2[i] in self._bowl_geom_ids:
                    return True
        return False
