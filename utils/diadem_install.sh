#!/bin/bash
virtualenv -p python3.7 ./python/venv
source ./bark_ml/python_wrapper/venv/bin/activate && pip3.7 install --no-cache-dir --upgrade --trusted-host pypi.org -r utils/docker/installers/requirements_diadem.txt 