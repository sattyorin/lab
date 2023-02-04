import numpy as np

NUM_SAMPLE = 9
NUM_INGREDIENTS = 6
FILE_PATH = "data/ideal_ingredient_positions"
ingredients_position = np.zeros((NUM_SAMPLE, NUM_INGREDIENTS, 2))

radius = 0.03
for n in range(NUM_SAMPLE):
    for i, j in enumerate([0, 2, 4, 1, 3, 5]):
        ingredients_position[n, i, 0] = radius * np.cos(
            2 * np.pi / NUM_INGREDIENTS * j
        )
        ingredients_position[n, i, 1] = radius * np.sin(
            2 * np.pi / NUM_INGREDIENTS * j
        )
np.save(FILE_PATH, ingredients_position)
