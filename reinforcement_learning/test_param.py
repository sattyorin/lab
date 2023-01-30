import envs
import gym
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    env = gym.make("stir-v0", render_mode="human")
    observation_space = env.observation_space
    action_space = env.action_space

    min_velocity = -0.08
    max_velocity = 0.08
    velocity_step = 0.0002
    action_sum_step = (
        np.sum(action_space.high) - np.sum(action_space.low)
    ) / 1000
    angle_step = 0.1

    velocity_xy_array = [
        v * velocity_step
        for v in range(
            int(min_velocity / velocity_step), int(max_velocity / velocity_step)
        )
    ]
    velocity_array = [
        np.linalg.norm(np.array([v, v])) for v in velocity_xy_array
    ]
    reward_small_velocity_array = [
        env.get_small_velocity_reward(v) for v in velocity_array
    ]
    angle_array = [
        a * angle_step
        for a in range(
            int(-np.pi / angle_step), int((np.pi + angle_step) / angle_step)
        )
    ]
    angular_acceleration = [
        env.get_angular_acceleration(1.0, 0.0, np.cos(a), np.sin(a))
        for a in angle_array
    ]
    action_array = [
        np.array([a * action_sum_step])
        for a in range(
            int(np.sum(action_space.low) / action_sum_step),
            int(np.sum(action_space.high) / action_sum_step),
        )
    ]
    action_array_positive = [
        np.array([a * action_sum_step])
        for a in range(250, int(np.sum(action_space.high) / action_sum_step))
    ]
    penalty_large_control = [
        env.get_large_control_penalty(a) for a in action_array_positive
    ]
    penalty_small_control = [
        env.get_small_control_penalty(a) for a in action_array_positive
    ]

    # plt.plot(velocity_array, reward_small_velocity_array)
    # plt.plot(angle_array, angular_acceleration)
    # plt.plot(action_array_positive, penalty_large_control)
    # plt.plot(action_array_positive, penalty_small_control)
    plt.show()
