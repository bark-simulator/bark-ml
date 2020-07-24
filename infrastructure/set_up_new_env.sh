#!/bin/zsh
env_name=$1

BASE_BRANCH="develop"
MARCO_CHECKPOINT_PATH="/Users/marco.oliva/Development/bark-ml_logs/checkpoints"
MARCO_SUMMARY_PATH="/Users/marco.oliva/Development/bark-ml_logs/summaries"
DOCKER_CHECKPOINT_PATH="/bark/checkpoints"
DOCKER_SUMMARY_PATH="/bark/summaries"
FILE_WITH_PATHS_TO_REPLACE="./examples/tfa_gnn.py"

git clone git@github.com:mrcoliva/bark-ml.git $env_name
cd $env_name
git checkout $BASE_BRANCH

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
'
cp default_config.py config.py

#sed "s;$MARCO_CHECKPOINT_PATH;$DOCKER_CHECKPOINT_PATH;g" $FILE_WITH_PATHS_TO_REPLACE > $FILE_WITH_PATHS_TO_REPLACE
#sed "s;$MARCO_SUMMARY_PATH;$DOCKER_SUMMARY_PATH;g" $FILE_WITH_PATHS_TO_REPLACE > $FILE_WITH_PATHS_TO_REPLACE
