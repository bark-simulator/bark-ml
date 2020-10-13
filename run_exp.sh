#!bin/bash
cd /bark-ml/bark-ml
virtualenv -p python3 ./bark_ml/python_wrapper/venv --system-site-packages
. ./bark_ml/python_wrapper/venv/bin/activate
bazel run //experiments:experiment_runner -- $@