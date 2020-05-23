#!/bin/bash
# $1 configuration, $2 user, $3 goal name
echo "Building $.."
bazel build //examples:diadem_dqn

echo "Uploading.."
rsync bazel-bin/examples/diadem_dqn* bernhard@8gpu:/mnt/glusterdata/home/$1/$2 -a --copy-links -v -z -P
rsync utils/cluster/run.sh bernhard@8gpu:/mnt/glusterdata/home/$1/$2/run.sh -a --copy-links -v -z -P

# ssh 8gpu && cd $3 && sbatch run.sh
# echo "Training started!"