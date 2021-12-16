# Experiment Runner

To ease tracking and training configurations, the experiment runner converts the json into a runnable experiment.
To avoid mismatching training and execution runs, the experiment runner additionally hashes the json-file (to provide reproducibility to some extent).

## Usage

First, open a terminal in the root folder of bark-ml. Then, execute `source utils/dev_into.sh`

To train a neural network that does not obey traffic rules (as reference):

`bazel run //experiments:experiment_runner --define ltl_rules=true --jobs 2 -- --exp_json=$HOME/Documents/bark/bark-ml/bark-ml/experiments/configs/rules/single_lane_ref.json --mode=train`

To train a neural network that obeys traffic rules:

`bazel run //experiments:experiment_runner --define ltl_rules=true --jobs 2 -- --exp_json=$HOME/Documents/bark/bark-ml/bark-ml/experiments/configs/rules/single_lane.json --mode=train`

Save all scenarios where the reference-net violates traffic rules or causes a collision:

`bazel run //experiments:experiment_runner --define ltl_rules=true --jobs 2 -- --exp_json=$HOME/Documents/bark/bark-ml/bark-ml/experiments/configs/rules/single_lane_ref2.json --mode=collisions`
The scenarios are saved as `$HOME/dump.json`.

Execute the second net in the saved scenarios, and get a list of all scenarios where it did _not_ violate traffic rules:

`bazel run //experiments:experiment_runner --define ltl_rules=true --jobs 2 -- --exp_json=$HOME/Documents/bark/bark-ml/bark-ml/experiments/configs/rules/single_lane.json --mode=validate`

Visualize scenarios with this command:

`bazel run //experiments:experiment_runner --define ltl_rules=true --jobs 2 -- --exp_json=$HOME/Documents/bark/bark-ml/bark-ml/experiments/configs/rules/single_lane.json --mode=visualize`
