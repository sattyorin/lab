import numpy as np
from envs.stir.i_stir_env import IStirEnv
from envs.stir.stir_env0 import StirEnv0
from envs.stir.stir_env_xy_position_ingredients8_stir_with_moving_ingredients import (
    StirEnvXYPositionIngredients8StirWithMovingIngredients,
)
from envs.stir.stir_env_xyz_position import StirEnvXYZPosition
from envs.stir.stir_env_xyz_position_ingredients8_keep_moving_ingredients import (
    StirEnvXYZPositionIngredients8KeepMovingIngredients,
)
from envs.stir.stir_env_xyz_position_ingredients8_move_tool import (
    StirEnvXYZPositionIngredients8MoveTool,
)
from envs.stir.stir_env_xyz_position_ingredients8_stir import (
    StirEnvXYZPositionIngredients8Stir,
)
from envs.stir.stir_env_xyz_position_ingredients8_stir_with_moving_ingredients import (
    StirEnvXYZPositionIngredients8StirWithMovingIngredients,
)
from envs.stir.stir_env_xyz_position_move_tool import StirEnvXYZPositionMoveTool
from envs.stir.stir_env_xyz_velocity_ingredient4 import (
    StirEnvXYZVelocityIngredient4,
)


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
    elif name == "StirEnvXYZPositionIngredients8Stir":
        return StirEnvXYZPositionIngredients8Stir(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    elif name == "StirEnvXYZPositionMoveTool":
        return StirEnvXYZPositionMoveTool(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    elif name == "StirEnvXYZVelocityIngredient4":
        return StirEnvXYZVelocityIngredient4(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    elif name == "StirEnvXYZPositionIngredients8MoveTool":
        return StirEnvXYZPositionIngredients8MoveTool(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    elif name == "StirEnvXYZPositionIngredients8KeepMovingIngredients":
        return StirEnvXYZPositionIngredients8KeepMovingIngredients(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    elif name == "StirEnvXYZPositionIngredients8StirWithMovingIngredients":
        return StirEnvXYZPositionIngredients8StirWithMovingIngredients(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    elif name == "StirEnvXYPositionIngredients8StirWithMovingIngredients":
        return StirEnvXYPositionIngredients8StirWithMovingIngredients(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    elif name == "StirEnvXYZPosition":
        return StirEnvXYZPosition(
            init_tool_pose, num_ingredients, check_collision_with_bowl
        )
    else:
        raise ValueError(f"the class {name} is not registered")
