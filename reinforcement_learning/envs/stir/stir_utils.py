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


def get_negative_exp(value: float, minimum: float = 1.0) -> float:
    return np.exp(np.log(0.01) * value / minimum)


def get_subtract_negative_exp_from_one(
    value: float, target: float = 1.0, target_score: float = 0.99
) -> float:
    return 1 - np.exp(np.log(1 - target_score) * value / target)


def get_sigmoid(value: float, maximum: float = 1.0) -> float:
    return 1 / (1 + np.exp(2.0 * np.log(0.01) * value / maximum - np.log(0.01)))


def get_inverted_sigmoid(value: float, minimum: float = 1.0) -> float:
    return 1 / (
        1 + np.exp(-2.0 * np.log(0.01) * value / minimum + np.log(0.01))
    )


def get_reward_keep_moving_end_effector(
    velocity: float, target_velocity: float
) -> float:
    return get_subtract_negative_exp_from_one(velocity, target_velocity)


def get_reward_distance_between_tool_and_ingredient(
    ingredient_position: np.ndarray,
    tool_position: np.ndarray,
    bowl_diameter: float,
) -> float:
    return get_negative_exp(
        np.linalg.norm(ingredient_position - tool_position), bowl_diameter
    )


def get_reward_array_keep_moving_ingredients(
    ingredient_positions: np.ndarray,
    previous_ingredient_position: np.ndarray,
    target_moving_distance: float,
) -> float:
    """
    arg:
        ingredient_positions: [[x0,y0],[x1,y1],...] or [[x0,y0,z0],[x1,y1,z0],...]
    return:
        float: reward array
    """
    return get_sigmoid(
        np.linalg.norm(
            ingredient_positions - previous_ingredient_position,
            axis=1,
        ),
        target_moving_distance,
    )


def get_distance_array_to_center_point(
    ingredient_positions: np.ndarray,
) -> np.ndarray:
    """
    arg:
        ingredient_positions: [[x0,y0],[x1,y1],...] or [[x0,y0,z0],[x1,y1,z0],...]
    return:
        float: distance array
    """
    return np.linalg.norm(ingredient_positions, axis=1)


def get_reward_array_distance_to_center_point(
    distance_array: np.ndarray, bowl_radius: float
) -> float:
    return get_negative_exp(distance_array, bowl_radius)


def get_distance_to_center_plane(
    ingredient_positions: np.ndarray, every_other_ingredients: bool
) -> float:
    # TODO(sara): return distance array
    """
    arg:
        ingredient_positions: [[x0,y0],[x1,y1],...] or [[x0,y0,z0],[x1,y1,z0],...]
    return:
        float: distance
    """
    num_ingredients = ingredient_positions.shape[0]
    if every_other_ingredients:
        return sum(
            ingredient_positions[::2][:, 0][
                ingredient_positions[::2][:, 0] > 0.0
            ]
        ) - sum(
            ingredient_positions[1::2][:, 0][
                ingredient_positions[1::2][:, 0] < 0.0
            ]
        )
    else:
        return sum(
            ingredient_positions[: (num_ingredients // 2)][:, 0][
                ingredient_positions[: (num_ingredients // 2)][:, 0] > 0.0
            ]
        ) - sum(
            ingredient_positions[(num_ingredients // 2) :][:, 0][
                ingredient_positions[(num_ingredients // 2) :][:, 0] < 0.0
            ]
        )


def get_reward_distance_to_center_plane(
    distance: float, bowl_radius_times_num_ingredients: float
):
    return get_negative_exp(distance, bowl_radius_times_num_ingredients)


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


def get_reward_distance_between_two_centroids(
    distance: float, bowl_diameter: float
) -> float:
    return get_negative_exp(distance, bowl_diameter)


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


def get_mean_variance_array_delaunay_distance(
    ingredient_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # TODO(sara): every_other_ingredients
    """
    arg:
        ingredient_positions: [[x0,y0],[x1,y1],...]
    return:
        float: distance
    """
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

    return (
        np.array([np.mean(distance1), np.mean(distance2)]),
        np.array([np.var(distance1), np.var(distance2)]),
    )


def get_reward_mean_variance_array_delaunay_distance(
    mean_array: np.ndarray,
    variance_array: np.ndarray,
    bowl_diameter: float,
    coefficient: float,
) -> Tuple[float, float]:
    return get_negative_exp(
        abs(mean_array[0] - mean_array[1]), bowl_diameter
    ), np.mean(get_negative_exp(variance_array, coefficient))


def get_penalty_large_control(action: np.ndarray) -> float:
    return np.sum(np.power(action, 2))  # 20


def get_penalty_small_control(action: np.ndarray) -> float:
    return np.reciprocal(np.sum(np.power(action, 2)))  # 14
