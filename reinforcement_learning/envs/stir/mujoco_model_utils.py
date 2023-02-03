import mujoco
import numpy as np


def get_num_ingredient(model: mujoco.MjModel) -> int:
    num_ingredients = 0
    for first_adr in model.name_bodyadr:
        adr = first_adr
        while model.names[adr] != 0:
            adr += 1
        if model.names[first_adr:adr].decode().count("ingredient") > 0:
            num_ingredients += 1
    return num_ingredients


def get_ingredient_ids(
    model: mujoco.MjModel, num_ingredients: int
) -> np.ndarray:
    return np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ingredient{i}")
            for i in range(num_ingredients)
        ]
    )


def get_tool_id(model: mujoco.MjModel) -> np.ndarray:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tools")


def get_ingredient_geom_ids(
    model: mujoco.MjModel, num_ingredients: int
) -> np.ndarray:
    return np.array(
        [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_ingredient{i}"
            )
            for i in range(num_ingredients + 1)
        ]
    )


def get_tool_geom_ids(model: mujoco.MjModel, num_tool_geom: int) -> np.ndarray:

    return np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_tool{i}")
            for i in range(num_tool_geom + 1)
        ]
    )


def get_bowl_geom_ids(model: mujoco.MjModel, num_bowl_geom: int) -> np.ndarray:
    return np.insert(
        np.array(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"geom_bowl_fragment{i}",
                )
                for i in range(1, num_bowl_geom + 1)
            ]
        ),
        0,
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "geom_bowl_bottom0",
        ),
    )
