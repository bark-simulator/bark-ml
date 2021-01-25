#!/bin/bash
virtualenv -p python3.7 ./bark_ml/python_wrapper/venv
source ./bark_ml/python_wrapper/venv/bin/activate && pip install --no-cache-dir --upgrade --trusted-host pypi.org -r ./utils/docker/installers/requirements.txt 
# cpu version
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html