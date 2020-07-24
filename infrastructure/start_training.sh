#!/bin/bash

docker run -it --gpus all \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v ~/.Xauthority:/home/root/.Xauthority \
 -v $(pwd):/bark \
--network='host' \
--env DISPLAY \
bark_ml_image bash -c '
#export CUDA_VISIBLE_DEVICES="";
bash utils/install.sh;
source utils/dev_into.sh;
pip install networkx tf2-gnn;
while true;
        do
        timeout 300s bazel run --jobs 24 //examples:tfa_gnn -- --mode=train;
        sleep 0.1;
done
'
