from typing import Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, cache_size: int):
        # self.cache_size = cache_size
        self._cache_observations = np.array([])
        self._cache_actions = np.array([])
        self._cache_rewards = np.array([])
        self._cache_next_observations = np.array([])

    def get_cache_rewards(self) -> np.ndarray:
        return self._cache_rewards

    def set_cache_rewards(self, rewards: np.ndarray) -> None:
        self._cache_rewards = rewards

    def store_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
    ) -> None:

        # if len(self.cache_size) > self.cache_size:
        # self.cache = np.delete(self.cache, 0)
        self._cache_observations = np.append(
            self._cache_observations, observation
        )
        self._cache_actions = np.append(self._cache_actions, action)
        self._cache_rewards = np.append(self._cache_rewards, reward)
        self._cache_next_observations = np.append(
            self._cache_next_observations, next_observation
        )

    def sample_random_minibatch(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        pass

    # TODO(sara): reset cache
