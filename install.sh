#!/bin/bash
virtualenv -p python3.6 ./python/venv
source ./python/venv/bin/activate && pip install --no-cache-dir --upgrade -r utils/requirements.txt