#!/bin/bash

python3 train_ddpg.py \
        --env stir_gazebo-v0 \
        --specialization StirEnvXYZVelocityIngredient4 \
        --xml ingredients8_toolxyz \
        --is_train_eval_env_identical True
