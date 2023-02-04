import numpy as np

NUM_SAMPLE = 9
ALPHA = 3
NUM_INGREDIENTS = sum([1, 3, 5] * ALPHA) - 1

FILE_PATH = "data/ideal_ingredient_positions"
RADIUS_INGREDIENTS = 0.013 / 2
ingredients_position = np.zeros((NUM_INGREDIENTS, 2))

i = 0
radius = 0.03
for t in [1, 3, 5]:
    radius = t * RADIUS_INGREDIENTS
    if t == 1:
        alpha = t * 2
    else:
        alpha = t * ALPHA
    for n in range(alpha):
        ingredients_position[i, 0] = radius * np.cos(2 * np.pi / (alpha) * n)
        ingredients_position[i, 1] = radius * np.sin(2 * np.pi / (alpha) * n)
        i += 1
ret = []
for _ in range(NUM_SAMPLE):
    index = np.arange(NUM_INGREDIENTS)
    np.random.shuffle(index)
    ret.append(ingredients_position[index])
np.save(FILE_PATH, np.array(ret))
