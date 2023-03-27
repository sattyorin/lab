import os

import envs.stir.stir_utils as stir_utils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy.spatial import Delaunay

RADIUS_BOWL = 0.04
RADIUS_INGREDIENTS = 0.013 / 2
NUM_INGREDIENTS = 6
MAX_TRIAL = 1000000
DIR_PATH = "data_eval_mixing"
SAMPLE_COLS = 6
SAMPLE_ROWS = 2
WEIGHT_DISTANCE = 1.0
WEIGHT_MEAN = 1.0
WEIGHT_VARIANCE = 1.0
TRIPLOT = True
CIRCLE = True

file_path = os.path.join(
    DIR_PATH,
    f"distance{WEIGHT_DISTANCE}_mean{WEIGHT_MEAN}_variance{WEIGHT_VARIANCE}_num{NUM_INGREDIENTS}",
)

if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH)

# get sample
if not os.path.isfile(f"{file_path}.npy"):
    sample = []
    for _ in range(MAX_TRIAL):
        radius = np.random.uniform(
            0.0, RADIUS_BOWL - RADIUS_INGREDIENTS, NUM_INGREDIENTS
        )
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
        if (distance_array > (RADIUS_INGREDIENTS * 2.0)).all():
            print("found")
            sample.append(ingredient_positions)
            if len(sample) == SAMPLE_COLS * SAMPLE_ROWS:
                break
    sample = np.array(sample)
    np.save(file_path, sample)
else:
    sample = np.load(f"{file_path}.npy")

# calculate reward
num_sample = sample.shape[0]
num_ingredients = sample[0].shape[0]

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
        reward_delaunay_mean,
        reward_delaunay_variance,
    ) = stir_utils.get_reward_mean_variance_array_delaunay_distance(
        mean_array,
        variance_array,
        RADIUS_BOWL * 2,
        RADIUS_BOWL**2,  # TODO(sara): tmp
    )

    sample_reward = np.append(
        sample_reward,
        reward_distance_between_two_centroids * WEIGHT_DISTANCE
        + reward_delaunay_mean * WEIGHT_MEAN
        + reward_delaunay_variance * WEIGHT_VARIANCE,
    )
    sample_reward_individual.append(
        np.array(
            [
                sample_reward[-1],
                reward_distance_between_two_centroids,
                reward_delaunay_mean,
                reward_delaunay_variance,
            ]
        ),
    )

# sort
sample_arg_sort = np.argsort(sample_reward)


# create graph
fig = plt.figure(figsize=(SAMPLE_COLS * 1.6, SAMPLE_ROWS * 1.6))
axes_array = [
    fig.add_subplot(SAMPLE_ROWS, SAMPLE_COLS, i + 1) for i in range(num_sample)
]

for i, index in enumerate(sample_arg_sort):
    ingredient_positions = sample[index]

    circle = patches.Circle(
        (0, 0),
        RADIUS_BOWL,
        facecolor="white",
        edgecolor="gray",
    )
    axes_array[i].add_patch(circle)

    if TRIPLOT:
        tri1 = Delaunay(ingredient_positions[: num_ingredients // 2])
        tri2 = Delaunay(ingredient_positions[num_ingredients // 2 :])
        simplices1 = stir_utils.get_deleted_line(
            ingredient_positions[: num_ingredients // 2], tri1
        )[1]
        simplices2 = stir_utils.get_deleted_line(
            ingredient_positions[num_ingredients // 2 :], tri2
        )[1]
        axes_array[i].triplot(
            ingredient_positions[: num_ingredients // 2, 0],
            ingredient_positions[: num_ingredients // 2, 1],
            simplices1,
            color="pink",
        )
        axes_array[i].triplot(
            ingredient_positions[num_ingredients // 2 :, 0],
            ingredient_positions[num_ingredients // 2 :, 1],
            simplices2,
            color="yellowgreen",
        )

    if CIRCLE:
        for position in ingredient_positions[: num_ingredients // 2]:
            circle = patches.Circle(
                (position[0], position[1]),
                RADIUS_INGREDIENTS,
                facecolor="lavenderblush",
                edgecolor="pink",
                linewidth=1.5,
            )
            axes_array[i].add_patch(circle)

        for position in ingredient_positions[num_ingredients // 2 :]:
            circle = patches.Circle(
                (position[0], position[1]),
                RADIUS_INGREDIENTS,
                facecolor="mintcream",
                edgecolor="yellowgreen",
                linewidth=1.5,
            )
            axes_array[i].add_patch(circle)

    axes_array[i].set_xlim(-RADIUS_BOWL - 0.001, RADIUS_BOWL + 0.001)
    axes_array[i].set_ylim(-RADIUS_BOWL - 0.001, RADIUS_BOWL + 0.001)
    reward_total = np.round(sample_reward_individual[index][0], 2)
    reward1 = np.round(sample_reward_individual[index][1], 2)
    reward2 = np.round(sample_reward_individual[index][2], 2)
    reward3 = np.round(sample_reward_individual[index][3], 2)
    axes_array[i].set_title(
        f"{reward_total}\n{reward1} : {reward2} : {reward3}"
    )
    axes_array[i].axis("off")

fig.tight_layout()
plt.show()
