#!/bin/bash


BAZEL_CACHE_DIR="/root/.cache"
EXTERNAL_BARK_CACHE_DIR="/bark/.cache"

set -x

while [[ "$#" -gt 0 ]]
do
    case $1 in
        -t|--timeout) timeout="$2"; shift;;
        -d|--devices) devices="$2"; shift;;
        -a|--agent) agent="$2"; shift;;
        -m|--mode) mode="$2"; shift;;
        -s|--sleep_time_start) sleep_time_start="$2"; shift;;
        --move_checkpoints) move_checkpoints=true;;
        *) echo Unknown parameter "$1"; exit 1;; 
    esac
    shift
done

echo timeout: $timeout
echo devices: $devices


prepend_command=""
if [[ "$move_checkpoints" == true ]]
then
    prepend_command=$prepend_command"move_checkpoints_func; "
fi

if [[ $timeout != "" ]]
then
    prepend_command=$prepend_command"timeout --foreground $timeout "
fi

if [[ -v devices ]]
then
    visible_devices_command='export CUDA_VISIBLE_DEVICES="'"$devices"'"';
fi

if [[ ! -v agent ]]
then
    agent="tfa_gnn"
fi

[[ -v mode ]] || mode="train"
[[ -v sleep_time_start ]] || sleep_time_start=1



docker run -it --gpus all \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v ~/.Xauthority:/home/root/.Xauthority \
 -v $(pwd):/bark \
--network='host' \
--env DISPLAY \
bark_ml_image bash -c '
time_last_start="-1";
sleep '"$sleep_time_start"';
time_to_sleep=1;
move_checkpoints_func() {
    archive_name="checkpoints_archive/archived_$(date +%s)"
    mkdir -p $archive_name
    [[ -d checkpoints ]] && mv checkpoints $archive_name
}
sleep_if_necessary() {
    time_difference=$(( $(date +%s) - $time_last_start ))    
    time_last_start=$(date +%s)
    echo time_difference: $time_difference
    if (( $time_difference < 60 ))
    then
        echo Sleeping...
        sleep $time_to_sleep
        time_to_sleep=$(( $time_to_sleep * 2 ))
    else
        time_to_sleep=1;
    fi
}
'"$visible_devices_command"'
trap exit INT;
if [[ ! -d '"EXTERNAL_BARK_CACHE_DIR"' ]]
then
    cp -r '"$BAZEL_CACHE_DIR"' '"$EXTERNAL_BARK_CACHE_DIR"';
fi
rm -r '"$BAZEL_CACHE_DIR"'
ln -s '"$EXTERNAL_BARK_CACHE_DIR"' '"$BAZEL_CACHE_DIR"';
source utils/dev_into.sh;
while true;
        do
        sleep_if_necessary
        '"$prepend_command"' bazel run --jobs 12 //examples:'"$agent"' -- --mode='"$mode"';
        sleep 0.1;
done
'
