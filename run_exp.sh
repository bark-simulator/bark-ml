#!bin/bash
cd /bark-ml/bark-ml
virtualenv -p python3 ./bark_ml/python_wrapper/venv --system-site-packages
export MPLBACKEND="agg"
. ./bark_ml/python_wrapper/venv/bin/activate
bazel run --jobs=4 //experiments:experiment_runner -- $@