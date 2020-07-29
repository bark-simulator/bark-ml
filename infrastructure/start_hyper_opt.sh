#!/bin/bash

set -x
trap cleanup EXIT

HYPER_OPT_START_COMMAND="./start_training_docker.sh --agent hyperparameter_optimization --move_checkpoints --timeout 12h"

pids_to_kill=""

launch_hyper_opt_instance() {
    kitty $HYPER_OPT_START_COMMAND &
    pids_to_kill="$pids_to_kill $!"
}

cleanup() {
    for pid in $pids_to_kill
    do
        kill $pid
    done
}

for i in {1..5}
do
    launch_hyper_opt_instance
done

sleep 1000000000
