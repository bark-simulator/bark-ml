
<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark_ml_logo.png" width="65%" alt="BARK-ML" />
</p>

# BARK-ML - Machine Learning for Autonomous Driving

![CI Build](https://github.com/bark-simulator/bark-ml/workflows/CI/badge.svg)
![CI-DIADEM Build](https://github.com/bark-simulator/bark-ml/workflows/CI-DIADEM/badge.svg)

BARK-ML provides <i>simple-to-use</i> [OpenAi-Gym](https://github.com/openai/gym) environments for several scenarios, such as highway driving, merging and intersections.
Additionally, BARK-ML integrates <i>state-of-the-art</i> machine learning libraries to learn driving behaviors for autonomous vehicles.

BARK-ML supported machine learning libraries:

* [TF-Agents](https://github.com/tensorflow/agents)
* [Baselines](https://github.com/openai/baselines) (Planned)
* [Diadem](https://github.com/juloberno/diadem)

Install BARK-ML using `pip install bark-ml`.

## Gym Environments

<p align="center">
<img src="https://github.com/bark-simulator/bark-ml/raw/master/docs/images/bark-ml.gif" alt="BARK-ML Highway" />
</p>

```python
import gym
import numpy as np
import bark_ml.environments.gym

# environment
env = gym.make("merging-v0")

env.reset()
done = False
while done is False:
  # random action
  action = np.random.uniform(
    low=np.array([-0.5, -0.1]), high=np.array([0.5, 0.1]), size=(2, ))
  # step the world
  observed_next_state, reward, done, info = env.step(action)
  print(f"Observed state: {observed_next_state}, Action: {action}, Reward: {reward}, Done: {done}")
```

Available environments:

* `highway-v0`: Continuous highway environment
* `highway-v1`: Discrete highway environment
* `merging-v0`: Continuous merging environment
* `merging-v1`: Discrete merging environment
* `intersection-v0`: Continuous intersection environment
* `intersection-v1`: Discrete intersection environment

## TF-Agents

SAC-Agent learning a merging scenario:

```python
import gym
from absl import app
from absl import flags
import os
os.environ['GLOG_minloglevel'] = '3' 

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
from bark.runtime.viewer.video_renderer import VideoRenderer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorSACAgent
from bark_ml.library_wrappers.lib_tf_agents.runners import SACRunner


params = ParameterServer()
params["World"]["remove_agents_out_of_map"] = True

# create environment
bp = ContinuousMergingBlueprint(params,
                                number_of_senarios=2500,
                                random_seed=0)
env = SingleAgentRuntime(blueprint=bp,
                         render=False)

# SAC-agent
sac_agent = BehaviorSACAgent(environment=env, params=params)
env.ml_behavior = sac_agent
runner = SACRunner(params=params,
                   environment=env,
                   agent=sac_agent)
runner.Train()
runner.Visualize(5)
```

## License

BARK-ML specific code is distributed under MIT License.
