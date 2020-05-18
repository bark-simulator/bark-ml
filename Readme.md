
<p align="center">
<img src="utils/bark_ml_logo.png" width="65%" alt="BARK-ML" />
</p>

## BARK-ML - Machine Learning for Autonomous Driving

![CI Build](https://github.com/bark-simulator/bark-ml/workflows/CI/badge.svg)

BARK-ML provides <i>simple-to-use</i> [OpenAi-Gym](https://github.com/openai/gym) environments for several scenarios, such as highway driving, merging and intersections.
Additionally, BARK-ML integrates <i>state-of-the-art</i> machine learning libraries to learn driving behaviors for autonomous vehicles.

BARK-ML supported machine learning libraries:

* [TF-Agents](https://github.com/tensorflow/agents)
* [Baselines](https://github.com/openai/baselines) (Planned)

### Quick-Start

First, install the virtual python enviornment by running `bash install.sh` and enter it with `source dev_into.sh`.

Continuous environment: `bazel run //examples:continuous_env`:
Discrete environment: `bazel run //examples:continuous_env`:
TF-Agents exanple: `bazel run //examples:tfa`:

Available environments:

* `highway-v0`: Continuous highway environment
* `highway-v1`: Discrete highway environment
* `merging-v0`: Continuous highway environment
* `merging-v1`: Discrete highway environment
