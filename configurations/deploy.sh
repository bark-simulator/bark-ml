#!/bin/bash
# $1 configuration
# $2 destination

echo "Building.."
bazel build //configurations/$1:configuration
# copied_version should be path to the user space on the cluster
echo "Uploading.."
mkdir $2
rsync bazel-bin/configurations/$1/ $2 -a --copy-links -v -z -P
echo "Uploaded!"
# sbatch and image should be on the cluster
# 1. ssh login
# 2. execute sbatch


