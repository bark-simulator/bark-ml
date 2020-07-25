#!/bin/bash
LOGS_DIR_TO_MOUNG="additional_runs"
LOGS_DIR_TENSORBOARD="/logs/symlinks"

docker run -it -p 6100:6006 -v $(pwd)/$LOGS_DIR_TO_MOUNG:/logs tensorflow/tensorflow tensorboard --bind_all --logdir $LOGS_DIR_TENSORBOARD
