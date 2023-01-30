import numpy as np
import tf


def get_euler_pose_from_quaternion(pose: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            pose[:3],
            tf.transformations.euler_from_quaternion(pose[3:]),
        ]
    )


def get_distance_between_two_centroids(
    ingredient_positions: np.ndarray, every_other_ingredients: bool
) -> float:
    """
    arg:
        ingredient_positions: [[x0,y0],[x1,y1],...] or [[x0,y0,z0],[x1,y1,z0],...]
    return:
        float: distance
    """
    num_ingredients = ingredient_positions.shape[0]
    if every_other_ingredients:
        centroid1 = np.mean(ingredient_positions[::2], axis=0)
        centroid2 = np.mean(ingredient_positions[1::2], axis=0)
    else:
        centroid1 = np.mean(
            ingredient_positions[: (num_ingredients // 2)], axis=0
        )
        centroid2 = np.mean(
            ingredient_positions[(num_ingredients // 2) :], axis=0
        )
    distance = np.linalg.norm(centroid1[0:2] - centroid2[0:2])

    return distance


def get_reward_stir(distance: float, stir_coefficient=1.0) -> float:
    """
    args:
        distance: get from get_distance_between_two_centroid
        stir_cofficient: the larger the value, the steeper the slope
    retrun:
        float: score
    """
    return np.exp(-stir_coefficient * distance)


def get_reward_small_velocity(velocity: float, target_velocity: float) -> float:
    return 1 - np.exp(-velocity / (target_velocity - target_velocity * 0.7))


def get_penalty_large_control(action: np.ndarray) -> float:
    return np.sum(np.power(action, 2))  # 20


def get_penalty_small_control(action: np.ndarray) -> float:
    return np.reciprocal(np.sum(np.power(action, 2)))  # 14


def get_distance_score(distance: float) -> float:
    return (1 + np.tanh((-abs(distance) * 200 + 5) / 2)) / 2


def get_reward_ingredient_movement(
    ingredient_position: np.ndarray, previous_ingredient_position: np.ndarray
) -> float:
    return (
        sum(
            np.linalg.norm(
                ingredient_position - previous_ingredient_position,
                axis=1,
            )
        )
        * 1000
    )
