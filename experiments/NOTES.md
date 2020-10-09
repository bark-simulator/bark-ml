# 1. build with bazel

{
    "blueprint": {
        "name": "ContinuousMergingBlueprint",
        "number_of_senarios": 2500,
        "random_seed": 1000
    },
    "observer": "GraphObserver",
    "evaluator": "GoalReached",
    "agent": {
        "name": "BehaviorGraphSACAgent",

    }
}

bp = ContinuousMergingBlueprint(params,
                                number_of_senarios=2500,
                                random_seed=0)
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