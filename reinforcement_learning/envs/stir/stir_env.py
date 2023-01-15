import os
import random
import sys
from typing import Dict, Tuple

import mujoco
import numpy as np
from gym import spaces, utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

_FRAME_SKIP = 1
_TIME_STEP = 0.01
_ENV = "stir-v0"
_TOOL_POSITION = "tool_position"
_TOOL_ORIENTATION = "tool_orientation"
_INGREDIENTS_POSE_INDEX = 0
_RESET_INGREDIENTS_RADIUS_MIN = 0.02
_RESET_INGREDIENTS_RADIUS_MAX = 0.04
_TOOL_GEOM_NUM = 2  # TODO(sara): get num from model
_BOWL_GEOM_NUM = 24  # TODO(sara): get num from model
_THRESHOLD_DISTANCE = 0.01
_LARGE_CONTROL_PENALTY_WEIGHT = 50.0
_SMALL_CONTROL_PENALTY_WEIGHT = 100.0
_TARGET_VELOCITY = 0.08
_RESET_NOISE_SCALE = 0.01


class StirEnv(MujocoEnv, utils.EzPickle):
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

        env_directory = os.path.abspath(os.path.dirname(__file__))
        xml_file_path: str = os.path.join(
            env_directory,
            "xmls",
            f"{_ENV}.xml",
        )
        mesh_dir_path: str = os.path.join(env_directory, "mesh")

        assets = dict()
        if os.path.exists(mesh_dir_path):
            mesh_files = os.listdir(mesh_dir_path)
            for file in mesh_files:
                with open(os.path.join(mesh_dir_path, file), "rb") as f:
                    assets[file] = f.read()

        self.model = mujoco.MjModel.from_xml_path(xml_file_path, assets)

        self.num_ingredients = 0
        for first_adr in self.model.name_bodyadr:
            adr = first_adr
            while self.model.names[adr] != 0:
                adr += 1
            if self.model.names[first_adr:adr].decode().count("ingredient") > 0:
                self.num_ingredients += 1

        self.tool_pose_index = (
            _INGREDIENTS_POSE_INDEX + self.num_ingredients * 7
        )

        self.ingredient_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"ingredient{i}"
                )
                for i in range(self.num_ingredients)
            ]
        )

        self.tool_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "tool"
        )

        self.ingredient_geom_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ingredient{i}_geom"
                )
                for i in range(self.num_ingredients + 1)
            ]
        )

        self.tool_geom_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"tool{i}_geom"
                )
                for i in range(_TOOL_GEOM_NUM + 1)
            ]
        )

        self.bowl_geom_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"bowl_geom{i}"
                )
                for i in range(1, _BOWL_GEOM_NUM + 1)
            ]
        )

        self.observation_size_tool = 0
        for first_adr in self.model.name_jntadr:
            adr = first_adr
            while self.model.names[adr] != 0:
                adr += 1
            if (
                self.model.names[first_adr:adr].decode().count(_TOOL_POSITION)
                > 0
            ):
                self.observation_size_tool += 2  # pos + vel
            elif (
                self.model.names[first_adr:adr]
                .decode()
                .count(_TOOL_ORIENTATION)
                > 0
            ):
                self.observation_size_tool += 2  # pos + vel

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_size_tool + self.num_ingredients * 3,),
            dtype=np.float64,
        )

        utils.EzPickle.__init__(**locals())

        MujocoEnv.__init__(
            self,
            xml_file_path,
            _FRAME_SKIP,
            observation_space=observation_space,
            **kwargs,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([0.07, 0.1]),
            dtype=np.float32,
        )

        self.total_reward = 0.0
        self.pre_velocity_x = 0.0
        self.pre_velocity_y = 0.0
        self.num_step = 0
        self.pre_angle = 0.0

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:

        ctrl = np.zeros(self.action_space.shape)
        ctrl[0] = action[0] * np.cos(self.pre_angle + action[1])
        ctrl[1] = action[0] * np.sin(self.pre_angle + action[1])
        self.pre_angle += action[1]
        self.do_simulation(ctrl, self.frame_skip)
        observation = self._get_observation()
        reward, terminated = self._get_reward(observation, action)
        info: Dict[str, str] = {}

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def reset_model(self) -> np.ndarray:

        if self.num_ingredients > 0:
            for _ in range(100):
                mujoco.mj_resetData(self.model, self.data)
                for i in range(self.num_ingredients):
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
                observation = self._get_observation()
                if (
                    self._get_distance_between_two_centroid(observation)
                    > _THRESHOLD_DISTANCE * 2.0
                ):
                    break

        else:
            mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[
            self.tool_pose_index : self.tool_pose_index
            + int(self.observation_size_tool / 2)
        ] = self.init_qpos[
            self.tool_pose_index : self.tool_pose_index
            + int(self.observation_size_tool / 2)
        ] + self.np_random.uniform(
            low=-_RESET_NOISE_SCALE,
            high=_RESET_NOISE_SCALE,
            size=int(self.observation_size_tool / 2),
        )
        self.data.qvel[
            self.tool_pose_index
            - self.num_ingredients : self.tool_pose_index
            - self.num_ingredients
            + int(self.observation_size_tool / 2)
        ] = self.init_qvel[
            self.tool_pose_index
            - self.num_ingredients : self.tool_pose_index
            - self.num_ingredients
            + int(self.observation_size_tool / 2)
        ] + self.np_random.uniform(
            low=-_RESET_NOISE_SCALE,
            high=_RESET_NOISE_SCALE,
            size=int(self.observation_size_tool / 2),
        )

        mujoco.mj_forward(self.model, self.data)
        observation = self._get_observation()

        self.total_reward = 0.0
        self.pre_velocity_x = 0.0
        self.pre_velocity_y = 0.0
        self.num_step = 0
        self.pre_angle = 0.0

        return observation

    def _get_observation(self) -> np.ndarray:

        position = self.data.qpos[
            self.tool_pose_index : self.tool_pose_index
            + int(self.observation_size_tool / 2)
        ].copy()
        velocity = self.data.qvel[
            self.tool_pose_index
            - self.num_ingredients : self.tool_pose_index
            - self.num_ingredients
            + int(self.observation_size_tool / 2)
        ].copy()

        ingredients_observation = np.array([])
        for i in range(self.num_ingredients):
            ingredients_observation = np.concatenate(
                [
                    ingredients_observation,
                    np.array(
                        self.data.qpos[
                            _INGREDIENTS_POSE_INDEX
                            + 7 * i : _INGREDIENTS_POSE_INDEX
                            + 7 * i
                            + 3
                        ]
                    ),
                ],
                dtype=float,
            )

        observation = np.concatenate(
            [position, velocity, ingredients_observation]
        )

        return observation

    def _get_distance_between_two_centroid(
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

    def get_angular_acceleration(
        self,
        velocity_x: float,
        velocity_y: float,
        pre_velocity_x: float,
        pre_velocity_y: float,
    ) -> float:
        inner = np.inner(
            np.array([velocity_x, velocity_y]),
            np.array([pre_velocity_x, pre_velocity_y]),
        )
        velocity = np.linalg.norm(np.array([velocity_x, velocity_y]))
        pre_velocity = np.linalg.norm(
            np.array([pre_velocity_x, pre_velocity_y])
        )

        if velocity == 0.0:
            velocity = sys.float_info.epsilon
        if pre_velocity == 0.0:
            pre_velocity = sys.float_info.epsilon

        return np.arccos(inner / (velocity * pre_velocity))

    def get_large_control_penalty(self, action: np.ndarray) -> float:
        return np.sum(np.power(action * _LARGE_CONTROL_PENALTY_WEIGHT, 14))

    def get_small_control_penalty(self, action: np.ndarray) -> float:
        return np.reciprocal(
            np.sum(np.power(action * _SMALL_CONTROL_PENALTY_WEIGHT, 20))
        )

    def _get_reward(
        self, observation: np.ndarray, action: np.ndarray
    ) -> Tuple[float, bool]:

        velocity_x = observation[self.observation_size_tool // 2]
        velocity_y = observation[self.observation_size_tool // 2 + 1]

        # collision detection
        # for i in range(self.data.contact.geom1.size):
        #     if self.data.contact.geom1[i] in self.bowl_geom_ids:
        #         if self.data.contact.geom2[i] in self.tool_geom_ids:
        #             return -100.0, True
        #     if self.data.contact.geom2[i] in self.tool_geom_ids:
        #         if self.data.contact.geom2[i] in self.bowl_geom_ids:
        #             return -100.0, True

        # drop detection
        # if (
        #     observation[self.observation_size_tool + 2] < 0.02
        #     and observation[self.observation_size_tool + 5] < 0.02
        # ):
        #     return -1.0, True

        # stir reward
        distance = 1 - self._get_distance_between_two_centroid(observation)
        reward_distance = 1 - distance
        if distance < _THRESHOLD_DISTANCE:
            print("done")
            return 100, True

        velocity = np.linalg.norm(np.array([velocity_x, velocity_y]))
        reward_small_velocity = self.get_small_velocity_reward(velocity)
        # angular_acceleration = self.get_angular_acceleration(
        #     velocity_x, velocity_y, self.pre_velocity_x, self.pre_velocity_y
        # )
        # penalty_large_control = self.get_large_control_penalty(action)
        # penalty_small_control = self.get_small_control_penalty(action)
        reward = reward_small_velocity + reward_distance
        self.total_reward += reward_small_velocity
        self.num_step += 1

        if self.total_reward / self.num_step < 0.8:
            return reward, True

        # self.pre_velocity_x = velocity_x
        # self.pre_velocity_y = velocity_y

        return reward, False
