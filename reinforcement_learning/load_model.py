import gym
import numpy as np

import envs

if __name__ == "__main__":
    env_id = "linear_actuator_array-v0"
    env = gym.make(env_id)

    env.reset()
    for _ in range(5):
        for _ in range(100):
            env.render()
            env.step(env.action_space.sample())
        env.reset()
