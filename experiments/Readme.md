# Experiment Runner

To ease tracking and training configurations, the experiment runner converts the json into a runnable experiment.
To avoid mismatching training and execution runs, the experiment runner additionally hashes the json-file (to provide reproducibility to some extent).

## Usage

For training, run the following command:
`bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/highway_gnn.json --mode=train`

To visualize the current checkpoint, run:
`bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/highway_gnn.json`

And to evaluate the performance of the agent, use:
`bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/highway_gnn.json --mode=evaluate`

## Cluster Usage

sbatch run_experiment_normal.sh --exp_json=experiments/experiment_runner.runfiles/bark_ml/experiments/configs/phd/01_hyperparams/dnns/merging_large_network.json --mode=train

## Run driving experiment
bazel run //experiments:experiment_runner --define build_platform=macos -- --exp_json=/Users/hart/Development/bark-ml/experiments/configs/driving/single_lane.json