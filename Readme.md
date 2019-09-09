# BARK Machine Learning (BARK-ML)

This repository is built upon BARK and TensorFlow-Agents to enable state-of-the-art reinforcement learning applied to autonomous driving.

Currently, we support the following agents in BARK-ML:

* Soft-Actor-Critic (SAC)

## HowTo

In order to be able to run various algorithms, first, install the virtual python environment by running `bash install.sh`.
Next, enter the created virtual environment with the command `source dev_into.sh`. In order to verify the include and functionalities of bark you can further run `bazel test //...`.