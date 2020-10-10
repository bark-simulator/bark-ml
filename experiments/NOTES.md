# 1. build with bazel
# 2. one js file that declares everything
# 3. create hash out of file and handle the training like this

--> load JSON FILE via ParameterServer -> will enable ConfigurableScenarioConf
experiment = Experiment("experiment_01.json")
experiment.train()
experiment.visualize()
experiment.evaluate()


{
    "blueprint": {
        "name": "ContinuousMergingBlueprint",
        "number_of_senarios": 2500,
        "random_seed": 1000
    },
    "observer": {
        "name": "GraphObserver",
        ...
    },
    "evaluator": {
        "name": "GoalReached",
        ...
    },
    "agent": {
        "name": "BehaviorGraphSACAgent",
        ...
    },
    "runner": {
        "name": abc
        ...
    },
    "runtime": {
        "name": "SingleAgentRuntime",
        ...
    },
    "logging": {
        "checkpoint_path": "./",
        "summary_path": "./",
    } 
}


# NOTE: IF WE USE THE CONFIGURABLE SCENARIO GENERATION; WOULD BE REPRODUCIBLE

bp = ContinuousMergingBlueprint(params)
observer = GraphObserver(params=params)
env = SingleAgentRuntime(
    blueprint=bp,
    observer=observer,
    render=False)
# behavior
sac_agent = BehaviorGraphSACAgent(environment=env,
                                  observer=observer,
                                  params=params,
                                  init_gnn=init_gcn)
env.ml_behavior = sac_agent
# runner
runner = SACRunner(params=params,
                    environment=env,
                    agent=sac_agent)