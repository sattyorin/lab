import gym

gym.envs.registration.register(
    id="linear_actuator_array-v0",
    entry_point="envs.linear_actuator_array.linear_actuator_array_env:LinearActuatorArrayEnv",
    max_episode_steps=1000,
)

gym.envs.registration.register(
    id="stir-v0",
    entry_point="envs.stir.stir_mujoco_env:StirMujocoEnv",
    max_episode_steps=1000,
)

gym.envs.registration.register(
    id="stir_gazebo-v0",
    entry_point="envs.stir.stir_gazebo_env:StirGazeboEnv",
    max_episode_steps=1000,
)
