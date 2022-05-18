# Experiment Runner

To ease tracking and training configurations, the experiment runner converts the json into a runnable experiment.
To avoid mismatching training and execution runs, the experiment runner additionally hashes the json-file (to provide reproducibility to some extent).

## Usage

For training, run the following command:
`bazel run //experiments:run_experiment -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/highway_gnn.json --mode=train`

To visualize the current checkpoint, run:
`bazel run //experiments:run_experiment -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/highway_gnn.json`

And to evaluate the performance of the agent, use:
`bazel run //experiments:run_experiment -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/highway_gnn.json --mode=evaluate`

## Cluster Usage

sbatch run_experiment_normal.sh --exp_json=experiments/run_experiment.runfiles/bark_ml/experiments/configs/phd/01_hyperparams/dnns/merging_large_network.json --mode=train

## Run driving experiment
bazel run //experiments:run_experiment --define build_platform=macos -- --exp_json=/Users/hart/Development/bark-ml/experiments/configs/driving/single_lane.json


sudo openvpn --config ~/.ovpn --auth-user-pass --auth-retry interact

First exp:
tmux new -d -s 01_single_lane

hart@fortiss-8gpu:~$ bash run_experiment_normal.sh --exp_json=./experiments/run_experiment.runfiles/bark_ml/experiments/configs/driving/single_lane_max_vel.json --mode=train

hart@fortiss-8gpu:~$ bash run_experiment_normal.sh --exp_json=./experiments/run_experiment.runfiles/bark_ml/experiments/configs/driving/single_lane_sparse.json --mode=train

tmux attach -t 01_single_lane

cntrl + b; then d



