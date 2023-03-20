#!/bin/bash

DIR=$(cd $(dirname $0); pwd)

# LATEST_FILE="`pwd`/`ls -lt *.txt | head -n 1 | gawk '{print $9}'`"

# python3 ../train_ddpg.py \
#         --load results/ee50e9abd77b44b94eb8a7d2c20ffdafd3810afc-8056d535-c33ee72c/1000000_finish/ \
#         --demo
