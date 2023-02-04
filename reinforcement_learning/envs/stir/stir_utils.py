from typing import Tuple

import numpy as np
import tf
from scipy.spatial import ConvexHull, Delaunay


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


def _get_delaunay_neighbour_indices(tri: Delaunay) -> np.ndarray:
    """
    Fine each pair of neighbouring vertices in the delaunay triangulation.
    :param vertices: The vertices of the points to perform Delaunay triangulation on
    :return: The pairs of indices of vertices
    """
    # tri = Delaunay(vertices)
    spacing_indices, neighbours = tri.vertex_neighbor_vertices
    ixs = np.zeros((2, len(neighbours)), dtype=int)
    np.add.at(
        ixs[0], spacing_indices[1 : int(np.argmax(spacing_indices))], 1
    )  # The argmax is unfortuantely needed when multiple final elements the same
    ixs[0, :] = np.cumsum(ixs[0, :])
    ixs[1, :] = neighbours
    # assert np.max(ixs) < len(vertices)
    return ixs


def get_distance_array_delaunay(ingredient_positions: np.ndarray) -> np.ndarray:
    tri = Delaunay(ingredient_positions)
    index1, index2 = _get_delaunay_neighbour_indices(tri)
    matrix = np.zeros((tri.npoints, tri.npoints), dtype=int)
    for i1, i2 in zip(index1, index2):
        matrix[i1, i2] = 1
    for i1, i2 in get_deleted_line(ingredient_positions, tri)[0]:
        matrix[i1, i2] = 0
        matrix[i2, i1] = 0
    index1, index2 = [], []
    for i in range(tri.npoints - 1):
        for j in range(i + 1, tri.npoints):
            if matrix[i, j] == 1:
                index1.append(i)
                index2.append(j)
    return np.linalg.norm(
        ingredient_positions[index1] - ingredient_positions[index2], axis=1
    )


def get_deleted_line(
    ingredient_positions: np.ndarray, tri: Delaunay
) -> Tuple[np.ndarray, np.ndarray]:
    # TODO(sara): chaos
    # tri = Delaunay(ingredient_positions)
    triangles = ingredient_positions[tri.simplices]
    if tri.nsimplex > tri.npoints - 2:
        max_angles_triangle = np.zeros(tri.nsimplex)
        max_angle_index_array_triangle = []
        for i_triangle in range(tri.nsimplex):
            triangle = triangles[i_triangle]
            vec = np.array(
                [
                    [
                        triangle[2][0] - triangle[1][0],
                        triangle[2][1] - triangle[1][1],
                    ],
                    [
                        triangle[2][0] - triangle[0][0],
                        triangle[2][1] - triangle[0][1],
                    ],
                    [
                        triangle[1][0] - triangle[0][0],
                        triangle[1][1] - triangle[0][1],
                    ],
                ]
            )
            norms = np.array(
                [
                    np.linalg.norm(vec[0]),
                    np.linalg.norm(vec[1]),
                    np.linalg.norm(vec[2]),
                ]
            )

            angles = np.zeros(3)
            index_array = np.array([[1, 2], [0, 2], [0, 1]])
            for i, index in enumerate(index_array):
                if i == 1:
                    inner = np.inner(vec[index[0]], -vec[index[1]])
                else:
                    inner = np.inner(vec[index[0]], vec[index[1]])
                cos_theta = inner / (norms[index[0]] * norms[index[1]])
                angles[i] = np.arccos(cos_theta)
            max_angle_index_array_arg = np.argmax(angles)
            max_angles_triangle[i_triangle] = angles[max_angle_index_array_arg]
            max_angle_index_array_triangle.append(
                index_array[max_angle_index_array_arg]
            )
        sort_angle_triangle_index = np.argsort(max_angles_triangle)
        ret = []
        sub_ret = []
        vertices = ConvexHull(ingredient_positions).vertices
        for triangle_index in sort_angle_triangle_index[::-1]:
            points = tri.simplices[triangle_index][
                max_angle_index_array_triangle[triangle_index]
            ]
            if not (points[0] in vertices and points[1] in vertices):
                continue
            ret.append(points)
            sub_ret.append(triangle_index)
            if tri.nsimplex - tri.npoints - 2 <= len(ret):
                break
        sub_ret = np.delete(tri.simplices, np.array(sub_ret), axis=0)
        return np.array(ret), sub_ret
    return np.array([]), tri.simplices


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
    distance1 = get_distance_array_delaunay(
        ingredient_positions[:harf_num_ingredients]
    )
    distance2 = get_distance_array_delaunay(
        ingredient_positions[harf_num_ingredients:]
    )
    # np.set_printoptions(suppress=True)
    # print(np.round(distance1, 9))
    # np.arctan2(np.sqrt(3), 1)

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
