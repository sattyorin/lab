#!/bin/bash

python3 train_ddpg.py \
        --env stir-v0 \
        --specialization StirEnvXYZPositionIngredients8Stir \
        --xml stir-ingredients8_toolxyz \
        --is_train_eval_env_identical False

