#!/bin/bash

git clone git@github.com:GAIL-4-BARK/large_data_store.git
unzip large_data_store/bark-ml.zip
mv bark-ml/expert_trajectories .
mv bark-ml/pretrained_agents .
rm -rf bark-ml
rm -rf large_data_store
