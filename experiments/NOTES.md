# 1. build with bazel
# 2. one js file that declares everything
# 3. create hash out of file and handle the training like this

--> load JSON FILE via ParameterServer -> will enable ConfigurableScenarioConf
experiment = ExperimentRunner("experiment_01.json")
experiment.train()
experiment.visualize()
experiment.evaluate()

# bazel run //experiments:runner --  --exp=experiments_01/02/03...  --mode=train


# NOTE: IF WE USE THE CONFIGURABLE SCENARIO GENERATION; WOULD BE REPRODUCIBLE

bp = ConfigurableScenarioBlueprint(params)  # eval
observer = GraphObserver(params=params)  # eval
env = SingleAgentRuntime(  # eval
    blueprint=bp,
    observer=observer,
    render=False)
# behavior
sac_agent = BehaviorGraphSACAgent(environment=env,  # eval
                                  observer=observer,
                                  params=params,
                                  init_gnn='init_gcn')  # eval
env.ml_behavior = sac_agent
# runner
runner = SACRunner(params=params,  # eval
                    environment=env,
                    agent=sac_agent)