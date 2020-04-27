# BARK-ML

<img src="docs/source/bark_ml_logo.png" width="65%" align="center" />

![CI Build](https://github.com/bark-simulator/bark-ml/workflows/CI/badge.svg)

Machine learning plays a vital role in decision making for autonomous agents. It enables learning based on experiences, a life-long improvement of the agents' behavior and much more.
With recent advances, especially, in the field of reinforcement learning great leaps in behavior generation of autonomous systems have been achieved.

BARK-ML takes state-of-the-art machine learning methods and applies these to a deterministic simulation of autonomous agents (BARK). The current agent models are based on the tf-agents library (https://github.com/tensorflow/agents). Due to the modular design and concept of BARK-ML, we hope to support a wider range of machine learning libraries in the future.

Reinforcement learning models that are currently available in BARK-ML:

* Soft-Actor-Critic (SAC)
* Proximal Policy Optimization (PPO)


## Quick-Start

To use BARK-ML a virtual python environment is recommended and can be installed using the command `bash install.sh`.
Next, to enter the created virtual environment run the command `source dev_into.sh`. To verify the functionality of BARK-ML run the tests with the command `bazel test //...`.

## Configurations (Getting Started)

Configurations are designed to run experiments in an hermetic container.

To train a configuration use the following command inside the virtual environment:

```
bazel run //configurations/highway:configuration -- --base_dir=/tmp/ --mode=train
```

There are three modes the configurations can be run with: `train`, `visualize` and `evaluate`. You can use these flags in the above stated bazel command.

Currently, you need to set the absolute path in the `config.json` for the checkpoints and summaries to work. You can visualize the training using tensoboard as follows: `tensorboard --logdir ./configurations/highway/summaries/`.
