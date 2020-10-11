--> load JSON FILE via ParameterServer -> will enable ConfigurableScenarioConf
experiment_runner = ExperimentRunner("experiment_01.json")
experiment_runner.train()
experiment_runner.visualize()
experiment_runner.evaluate()

# bazel run //experiments:runner --  --exp=experiments_01/02/03...  --mode=train
# NOTE: IF WE USE THE CONFIGURABLE SCENARIO GENERATION; WOULD BE REPRODUCIBLE
# NOTE: hash(Experiment, ML)

