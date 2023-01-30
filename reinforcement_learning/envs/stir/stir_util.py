from typing import Tuple

import numpy as np
import tf
from scipy.spatial import Delaunay, delaunay_plot_2d


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


def _get_delaunay_neighbour_indices(
    vertices: np.ndarray,
) -> np.ndarray:
    """
    Fine each pair of neighbouring vertices in the delaunay triangulation.
    :param vertices: The vertices of the points to perform Delaunay triangulation on
    :return: The pairs of indices of vertices
    """
    tri = Delaunay(vertices)
    spacing_indices, neighbours = tri.vertex_neighbor_vertices
    ixs = np.zeros((2, len(neighbours)), dtype=int)
    np.add.at(
        ixs[0], spacing_indices[1 : int(np.argmax(spacing_indices))], 1
    )  # The argmax is unfortuantely needed when multiple final elements the same
    ixs[0, :] = np.cumsum(ixs[0, :])
    ixs[1, :] = neighbours
    assert np.max(ixs) < len(vertices)
    return ixs


def get_score_delaunay(ingredient_positions: np.ndarray) -> Tuple[float, float]:
    harf_num_ingredients = ingredient_positions.shape[0] // 2
    index1, index2 = _get_delaunay_neighbour_indices(
        ingredient_positions[:harf_num_ingredients]
    )
    distance1 = np.linalg.norm(
        ingredient_positions[index1] - ingredient_positions[index2], axis=1
    )
    index1, index2 = _get_delaunay_neighbour_indices(
        ingredient_positions[harf_num_ingredients:]
    )
    distance2 = np.linalg.norm(
        ingredient_positions[harf_num_ingredients + index1]
        - ingredient_positions[harf_num_ingredients + index2],
        axis=1,
    )
    distance_meam_score1 = np.mean(distance1)
    distance_meam_score2 = np.mean(distance2)

    variance1 = np.var(distance1)
    variance2 = np.var(distance2)
