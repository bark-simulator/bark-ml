# Experiment Runner
To ease tracking and training configurations, the experiment runner converts the json into a runnable experiment.
To avoid mismatching training and execution runs, the experiment runner additionally hashes the json-file (to provide reproducibility to some extent).

## Usage
For training, run the following command:
`bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/highway_interaction_network.json --mode=train`

To visualize the current checkpoint, run:
`bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/highway_interaction_network.json`

And to evaluate the performance of the agent, use:
`bazel run //experiments:experiment_runner -- --exp_json=/ABSOLUTE_PATH/bark-ml/experiments/configs/highway_interaction_network.json --mode=evaluate`


## Notes
e.g.:
sbatch run_experiment_normal.sh --exp_json=experiments/experiment_runner.runfiles/bark_ml/experiments/configs/phd/01_hyperparams/dnns/merging_large_network.json --mode=train
sbatch run_experiment_lowprio.sh --exp_json=experiments/experiment_runner.runfiles/bark_ml/experiments/configs/phd/01_hyperparams/dnns/merging_small_network.json --mode=train
sbatch run_experiment_normal.sh --exp_json=experiments/experiment_runner.runfiles/bark_ml/experiments/configs/phd/01_hyperparams/gnns/merging_large_embedding.json --mode=train
sbatch run_experiment_lowprio.sh --exp_json=experiments/experiment_runner.runfiles/bark_ml/experiments/configs/phd/01_hyperparams/gnns/merging_small_embedding.json --mode=train
