
<p align="center">
<img src="docs/images/bark_ml_logo.png" width="65%" alt="BARK-ML" />
</p>

# BARK-ML - Machine Learning for Autonomous Driving

![CI Build](https://github.com/bark-simulator/bark-ml/workflows/CI/badge.svg)

BARK-ML provides <i>simple-to-use</i> [OpenAi-Gym](https://github.com/openai/gym) environments for several scenarios, such as highway driving, merging and intersections.
Additionally, BARK-ML integrates <i>state-of-the-art</i> machine learning libraries to learn driving behaviors for autonomous vehicles.

BARK-ML supported machine learning libraries:

* [TF-Agents](https://github.com/tensorflow/agents)
* [Baselines](https://github.com/openai/baselines) (Planned)

## Gym Environments

Bef, install the virtual python enviornment (`bash install.sh`) and enter it (`source dev_into.sh`).

Continuous environments with random actions: `bazel run //examples:continuous_env`
<p align="center">
<img src="docs/images/bark-ml.gif" alt="BARK-ML Highway" />
</p>

Available environments:

* `highway-v0`: Continuous highway environment
* `highway-v1`: Discrete highway environment
* `merging-v0`: Continuous highway environment
* `merging-v1`: Discrete highway environment

## TF-Agents

TF-Agents example (trained 15.000 episodes): `bazel run //examples:tfa`.

<p align="center">
<img src="docs/images/bark_ml_highway.gif" alt="BARK-ML Highway" />
</p>


## License

BARK-ML specific code is distributed under MIT License.
