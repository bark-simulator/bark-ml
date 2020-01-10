#!/bin/bash
# $1 configuration
# $2 user
# $3 goal name

echo "Building.."
bazel build //configurations/$1:configuration
# copied_version should be path to the user space on the cluster
echo "Uploading.."
rsync bazel-bin/configurations/$1/ 8gpu:/mnt/glusterdata/home/$2/$3 -a --copy-links -v -z -P
echo "Uploaded!"
# sbatch and image should be on the cluster
# 1. ssh login
# 2. execute sbatch
