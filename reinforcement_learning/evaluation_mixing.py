import os

import envs.stir.stir_utils as stir_utils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy.spatial import Delaunay

RADIUS_BOWL = 0.04
RADIUS_INGREDIENTS = 0.013 / 2
RADIUS_MIN = 0.0
# RADIUS_MAX = RADIUS_BOWL - RADIUS_INGREDIENTS * np.sqrt(2.0)
RADIUS_MAX = RADIUS_BOWL - RADIUS_INGREDIENTS
NUM_INGREDIENTS = 8
MAX_TRIAL = 1000000
# FILE_PATH = "data/ingredient_positions"
FILE_PATH = "data/ideal_ingredient_positions"
NUM_SAMPLE = 12

if not os.path.isfile(f"{FILE_PATH}.npy"):
    sample = []
    for _ in range(MAX_TRIAL):
        radius = np.random.uniform(RADIUS_MIN, RADIUS_MAX, NUM_INGREDIENTS)
        angle = np.random.uniform(0.0, 2.0 * np.pi, NUM_INGREDIENTS)

        ingredient_positions = np.vstack(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
            ]
        ).transpose()

        distance_array = stir_utils.get_distance_array_delaunay(
            ingredient_positions
        )
        # if (distance_array > (RADIUS_INGREDIENTS * np.sqrt(2.0) * 2.0)).all():
        if (distance_array > (RADIUS_INGREDIENTS * 2.0)).all():
            print("found")
            sample.append(ingredient_positions)
            if len(sample) == NUM_SAMPLE:
                break
    np.save(FILE_PATH, sample)
else:
    sample = np.load(f"{FILE_PATH}.npy")

num_sample = sample.shape[0]
num_ingredients = sample[0].shape[0]

# sample[:] = sample[2]

sample_reward = np.array([])
sample_reward_individual = []
for ingredient_positions in sample:
    distance_between_two_centroids = (
        stir_utils.get_distance_between_two_centroids(
            ingredient_positions,
            every_other_ingredients=False,
        )
    )
    reward_distance_between_two_centroids = (
        stir_utils.get_reward_distance_between_two_centroids(
            distance_between_two_centroids,
            RADIUS_BOWL * 2.0,
        )
    )

    (
        mean_array,
        variance_array,
    ) = stir_utils.get_mean_variance_array_delaunay_distance(
        ingredient_positions
    )

    (
        reward_dyelauna_mean,
        reward_dyelauna_variance,
    ) = stir_utils.get_reward_mean_variance_array_delaunay_distance(
        mean_array,
        variance_array,
        RADIUS_BOWL * 2,
        RADIUS_BOWL**2,  # TODO(sara): tmp
    )

    reward_distance_between_two_centroids *= 0.7
    sample_reward = np.append(
        sample_reward,
        reward_distance_between_two_centroids
        + reward_dyelauna_mean
        + reward_dyelauna_variance,
    )
    sample_reward_individual.append(
        np.array(
            [
                sample_reward[-1],
                reward_distance_between_two_centroids,
                reward_dyelauna_mean,
                reward_dyelauna_variance,
            ]
        ),
    )
# -------------------------------------------------------


sample_arg_sort = np.argsort(sample_reward)

fig = plt.figure(figsize=(9, num_sample))  # needs adjustment
axes_array = [
    fig.add_subplot(num_sample // 3, 3, i + 1) for i in range(num_sample)
]

for i, index in enumerate(sample_arg_sort):
    ingredient_positions = sample[index]

    circle = patches.Circle(
        (0, 0),
        0.04,
        facecolor="white",
        edgecolor="gray",
    )
    axes_array[i].add_patch(circle)

    tri1 = Delaunay(ingredient_positions[: num_ingredients // 2])
    tri2 = Delaunay(ingredient_positions[num_ingredients // 2 :])
    simplices1 = stir_utils.get_deleted_line(
        ingredient_positions[: num_ingredients // 2], tri1
    )[1]
    simplices2 = stir_utils.get_deleted_line(
        ingredient_positions[num_ingredients // 2 :], tri2
    )[1]
    # axes_array[i].triplot(
    #     ingredient_positions[: num_ingredients // 2, 0],
    #     ingredient_positions[: num_ingredients // 2, 1],
    #     tri1.simplices,
    #     # simplices1,
    #     color="red",
    # )
    # axes_array[i].triplot(
    #     ingredient_positions[num_ingredients // 2 :, 0],
    #     ingredient_positions[num_ingredients // 2 :, 1],
    #     tri2.simplices,
    #     # simplices2,
    #     color="green",
    # )
    axes_array[i].triplot(
        ingredient_positions[: num_ingredients // 2, 0],
        ingredient_positions[: num_ingredients // 2, 1],
        # tri1.simplices,
        simplices1,
        color="pink",
    )
    axes_array[i].triplot(
        ingredient_positions[num_ingredients // 2 :, 0],
        ingredient_positions[num_ingredients // 2 :, 1],
        # tri2.simplices,
        simplices2,
        color="yellowgreen",
    )

    # for position in ingredient_positions[: num_ingredients // 2]:
    #     circle = patches.Circle(
    #         (position[0], position[1]),
    #         RADIUS_INGREDIENTS,
    #         facecolor="white",
    #         edgecolor="pink",
    #     )
    #     axes_array[i].add_patch(circle)

    # for position in ingredient_positions[num_ingredients // 2 :]:
    #     circle = patches.Circle(
    #         (position[0], position[1]),
    #         RADIUS_INGREDIENTS,
    #         facecolor="white",
    #         edgecolor="yellowgreen",
    #     )
    #     axes_array[i].add_patch(circle)

    axes_array[i].set_xlim(-RADIUS_BOWL, RADIUS_BOWL)
    axes_array[i].set_ylim(-RADIUS_BOWL, RADIUS_BOWL)
    axes_array[i].set_title(f"{np.round(sample_reward_individual[index], 2)}")

fig.tight_layout()
plt.show()
