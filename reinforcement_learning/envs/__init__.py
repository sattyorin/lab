import gym

gym.envs.registration.register(
    id="linear_actuator_array-v0",
    entry_point="envs.linear_actuator_array.linear_actuator_array_env:LinearActuatorArrayEnv",
    max_episode_steps=1000,
)
