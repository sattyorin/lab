import gym
import numpy as np

import envs

if __name__ == "__main__":
    env_id = "linear_actuator_array-v0"
    env = gym.make(env_id)

    env.reset()
    action1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    action0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for _ in range(5):
        for _ in range(100):
            env.render()
            env.step(action1)
        for _ in range(100):
            env.render()
            env.step(action0)
            # env.step(env.action_space.sample())
        env.reset()
