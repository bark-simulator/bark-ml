# BARK-ML

<img src="docs/source/bark_ml_logo.png" width="65%" align="center" />

Machine learning plays a vital role in decision making for autonomous agents. It enables learning based on experiences, a life-long improvement of the agents' behavior and much more.
With recent advances, especially, in the field of reinforcement learning great leaps in behavior generation of autonomous systems have been achieved.

BARK-ML takes state-of-the-art machine learning methods and applies these to a deterministic simulation of autonomous agents (BARK). The current agent models are based on the tf-agents library (https://github.com/tensorflow/agents). Due to the modular design and concept of BARK-ML, we hope to support a wider range of machine learning libraries in the future.


Reinforcement learning models that are currently available in BARK-ML:

* Soft-Actor-Critic (SAC)

## HowTo
In order to use BARK-Ml a virtual python environment is recommended and can be installed using the command `bash install.sh`.
Next, to enter the created virtual environment run the command `source dev_into.sh`. Now, bazel build commands can be executed, such as `bazel test //...`.
