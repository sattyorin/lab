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

# TODO(sara): execute script
python3 train_ddpg.py \
        --load $LATEST_DIRECTORY \
        --demo
