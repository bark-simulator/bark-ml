#!/bin/bash
# $1 user
echo "Building"
bazel build //experiments:run_experiment

echo "Uploading.."
rsync bazel-bin/experiments/ 8gpu:/mnt/glusterdata/home/$1/experiments/ -a --copy-links -v -z -P
rsync ./experiments/run_experiment_normal.sh 8gpu:/mnt/glusterdata/home/$1/run_experiment_normal.sh -a --copy-links -v -z -P
rsync ./experiments/run_experiment_lowprio.sh 8gpu:/mnt/glusterdata/home/$1/run_experiment_lowprio.sh -a --copy-links -v -z -P