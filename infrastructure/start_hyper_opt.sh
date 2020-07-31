#!/bin/bash

trap cleanup EXIT

HYPER_OPT_START_COMMAND="./start_training_docker.sh --devices 1 --agent hyperparameter_optimization --move_checkpoints --timeout 12h"

pids_to_kill=""

launch_hyper_opt_instance() {
    sleep_time=$1
    command_to_run="$HYPER_OPT_START_COMMAND --sleep_time_start $sleep_time"
    kitty $command_to_run &
    pids_to_kill="$pids_to_kill $!"
}

cleanup() {
    for pid in $pids_to_kill
    do
        kill $pid
    done
}

for i in {1..4}
do
    sleep_time=$(( ($i - 1) * 30 ))
    launch_hyper_opt_instance $sleep_time
done

start_time=$(date +%s)

while true
do
    uptime_in_seconds=$(( $(date +%s) - $start_time ))
    uptime_in_hours=$(( $uptime_in_seconds / 3600 ))
    echo "Uptime: "$uptime_in_hours"h"
    sleep 1
done
