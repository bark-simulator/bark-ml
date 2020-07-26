#!/bin/bash

set -x

while [[ "$#" -gt 0 ]]
do
    case $1 in
        -t|--timeout) timeout="$2"; shift;;
        -d|--devices) devices="$2"; shift;;
        -a|--agent) agent="$2"; shift;;
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

docker run -it --gpus all \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v ~/.Xauthority:/home/root/.Xauthority \
 -v $(pwd):/bark \
--network='host' \
--env DISPLAY \
bark_ml_image bash -c '
'"$visible_devices_command"'
trap exit INT;
mv /root/.cache /bark/.cache;
ln -s /bark/.cache /root/.cache;
source utils/dev_into.sh;
while true;
        do
        '"$prepend_command"' bazel run --jobs 12 //examples:'"$agent"' -- --mode=train;
        sleep 0.1;
done
'
