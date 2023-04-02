#!/bin/bash

cd $(dirname $0)/..

LATEST_FILE=$(ls -t results| head -n 1)
echo $LATEST_FILE

LATEST_DIRECTORY=$(find results/$LATEST_FILE \
                    -mindepth 1 \
                    -maxdepth 1 \
                    -type d \
                    -printf '%T@ %p\n' | sort -n -r | head -n 1 | awk '{print $2}')
echo $LATEST_DIRECTORY
# TODO(sara): empty check LATEST_DIRECTORY

env=$(jq -r '.env' results/$LATEST_FILE/args.txt)
specialization=$(jq -r '.specialization' results/$LATEST_FILE/args.txt)
xml=$(jq -r '.xml' results/$LATEST_FILE/args.txt)
is_train_eval_env_identical=$(jq -r '.is_train_eval_env_identical' results/$LATEST_FILE/args.txt)

python3 train_ddpg.py \
        --env $env \
        --specialization $specialization  \
        --xml $xml \
        --is_train_eval_env_identical $is_train_eval_env_identical \
        --load $LATEST_DIRECTORY \
        --demo
