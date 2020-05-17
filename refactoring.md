# Goals Refactoring

- Better usability with other libraries
- Easy access to standard training environments (including meaningful observation, action spaces and reward settings for these environments)
- Existing configuration options still be possible
- Trained agents can be reused in scenarios as behavior models
- benchmark running over agent models
- maybe: training agents over benchmark database


# End User Interfaces

runtime_rl = blueprints.filter("highway", "dense", "macro_actions)

RuntimeRL(behavior_ml= BehaviorMacroActions(),)

# Internal Interfaces


## 1 BarkMLBehavior and Runtime Definition (DONE)

BarkMLBehavior(BehaviorModel):

    def __init__(action_definition, )

    self.observer = None
    self.action_definition = BehaviorMacroActions()

    def ActionToBehavior(boost:variant<> action):

    def plan(observed_world):
        self.action_definition.plan()

    def _train():


Constructors
    RunTimeRL(bark_ml_agent, evaluator, scenario_generation)
    RunTimeRL(blueprint)

    # has to call plan method



## 2 Setup some blueprints (DONE)

dir_gym_blueprints:
    blue_print_gym.py =

    read_blueprint.py



class BluePrint:
    self.evaluator
    self.scenario_generation
    self.bark_ml_agent # actions, observations

Gym environments with blueprints

behavior_uct = blueprints.filter("uct", "hypothesis", "iteration=2000")
bark_sac = blue

gym_env = blueprints.filter("gym",)

## 3 Check other libraries apart from tf agents
-> baselines, tf-agents (DONE), diadem


## 4 BarkML runner generate statistics and reusable models
--> integrated tf_agents runner
BarkMlRun
    self.sucessful_scenario_ids =[]
    self.sceanrio_param_file_trained
BarkMLBehaviorSACAgent(BARKMLBehavior):

    def __init__():

        self.checkpoint_path
        #load ...

    def plan(observed_world):
        nn_state = slf.observer.observe



# Timeplan / Tasks