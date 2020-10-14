#!bin/bash
cd /bark-ml/bark-ml
virtualenv -p python3 ./bark_ml/python_wrapper/venv --system-site-packages
export MPLBACKEND="agg"
. ./bark_ml/python_wrapper/venv/bin/activate
bazel --local_cpu_resources=4 run //experiments:experiment_runner -- $@