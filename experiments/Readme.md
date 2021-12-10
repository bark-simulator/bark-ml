# Experiment Runner

To ease tracking and training configurations, the experiment runner converts the json into a runnable experiment.
To avoid mismatching training and execution runs, the experiment runner additionally hashes the json-file (to provide reproducibility to some extent).

## Usage

To train a neural network that does not obey traffic rules (as reference):
`bazel run //experiments:experiment_runner --define ltl_rules=true --jobs 2 -- --exp_json=/home/groetzner/Documents/bark/bark-ml/bark-ml/experiments/configs/rules/single_lane_ref.json --mode=train`

To train a neural network that obeys traffic rules:
`bazel run //experiments:experiment_runner --define ltl_rules=true --jobs 2 -- --exp_json=/home/groetzner/Documents/bark/bark-ml/bark-ml/experiments/configs/rules/single_lane.json --mode=train`

Save all scenarios where the reference-net violates traffic rules:
`bazel run //experiments:experiment_runner --define ltl_rules=true --jobs 2 -- --exp_json=/home/groetzner/Documents/bark/bark-ml/bark-ml/experiments/configs/rules/single_lane_ref.json --mode=collisions`
The scenarios are saved as `$HOME/dump.json`.

Execute the second net in the saved scenarios:
`bazel run //experiments:experiment_runner --define ltl_rules=true --jobs 2 -- --exp_json=/home/groetzner/Documents/bark/bark-ml/bark-ml/experiments/configs/rules/single_lane.json --mode=validate`
