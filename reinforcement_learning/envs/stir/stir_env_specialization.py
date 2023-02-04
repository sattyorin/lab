import numpy as np
from envs.stir.i_stir_env import IStirEnv
from envs.stir.stir_env0 import StirEnv0
from envs.stir.stir_env_xyz_ingredients8_stir import StirEnvXYZIngredients8Stir


def get_specialization(
    name: str,
    init_tool_pose: np.ndarray,
    num_ingredients: int,
    check_collision_with_bowl,
) -> IStirEnv:
    if name == "StirEnv0":
        return StirEnv0(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    if name == "StirEnvXYZIngredients8Stir":
        return StirEnvXYZIngredients8Stir(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    else:
        raise ValueError(f"the class {name} is not registered")
