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
        *) echo Unknown parameter "$1"; exit 1;; 
    esac
    shift
done

echo timeout: $timeout
echo devices: $devices

prepend_command=""
if [[ $timeout != "" ]]
then
    prepend_command=$prepend_command"timeout $timeout "
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

docker run -it --gpus all \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v ~/.Xauthority:/home/root/.Xauthority \
 -v $(pwd):/bark \
--network='host' \
--env DISPLAY \
bark_ml_image bash -c '
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
        '"$prepend_command"' bazel run --jobs 12 //examples:'"$agent"' -- --mode='"$mode"';
        sleep 0.1;
done
'
