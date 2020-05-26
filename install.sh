#!/bin/bash
virtualenv -p python3.7 ./python/venv
source ./python/venv/bin/activate && pip3.7 install --no-cache-dir --upgrade --trusted-host pypi.org -r utils/docker/installers/requirements.txt 