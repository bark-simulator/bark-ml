#!/bin/bash
# $1 configuration, $2 user, $3 goal name
echo "Building $.."
bazel build //configurations/$1:configuration
echo "Uploading.."
# rsync bazel-bin/configurations/$1/ 8gpu:/mnt/glusterdata/home/$2/$3 -a --copy-links -v -z -P
rsync bazel-bin/configurations/$1/ ../experiment -a --copy-links -v -z -P
echo "Uploaded!"
