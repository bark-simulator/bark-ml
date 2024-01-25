# Copyright (c) 2020 Patrick Hart, Julian Bernhard,
# Klemens Esterle, Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from bark.core.world.evaluation import \
  EvaluatorGoalReached, EvaluatorCollisionEgoAgent, \
  EvaluatorStepCount, EvaluatorDrivableArea
from bark.core.geometry import Point2d
from bark_ml.evaluators.general_evaluator import *
from bark_ml.evaluators.stl.evaluator_stl import *
from bark_ml.evaluators.stl.label_functions.safe_distance_label_function import *

class GoalReached(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["GoalReachedEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "goal_functor": GoalFunctor(self._params)
      })


class RewardShapingGoalDistEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["RewardShapingGoalDistEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_goal_center_functor": PotentialGoalCenterlineFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "goal_functor": GoalFunctor(self._params)
      })
class TestRewardShapingGoalDistEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["TestRewardShapingGoalDistEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_drivable_area_functor" : CollisionDrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_center_functor": PotentialGoalCenterlineFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "goal_functor": GoalFunctor(self._params)
      })
class RewardShapingEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["RewardShapingEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_center_functor": PotentialCenterlineFunctor(self._params),
        "pot_vel_functor": PotentialVelocityFunctor(self._params),
        "goal_functor": GoalFunctor(self._params)
      })

class SimpleSingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["SimpleSingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_goal_center_functor": PotentialGoalCenterlineFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params)
      })
class TestSimpleSingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["TestSimpleSingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_drivable_area_functor" : CollisionDrivableAreaFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "pot_center_functor": PotentialGoalCenterlineFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "pot_goal_vel_functor" : PotentialGoalReachedVelocityFunctor(self._params)
      })
class SingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["SingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "pot_center_functor": PotentialCenterlineFunctor(self._params),
        "pot_vel_functor": PotentialVelocityFunctor(self._params)
      })

class RewardShapingEvaluatorMaxSteps(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["RewardShapingEvaluatorMaxSteps"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "pot_center_functor": PotentialCenterlineFunctor(self._params),
        "max_step_count_as_goal_functor": MaxStepCountAsGoalFunctor(self._params),
        "pot_vel_functor": PotentialVelocityFunctor(self._params)
      })
    
class SmoothnessSingleLaneEvaluator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params["ML"]["SmoothnessSingleLaneEvaluator"]
    super().__init__(
      params=self._params,
      bark_ml_eval_fns={
        "collision_functor" : CollisionFunctor(self._params),
        "low_speed_goal_reached_functor" : LowSpeedGoalFunctor(self._params),
        "drivable_area_functor" : DrivableAreaFunctor(self._params),
        "smoothness_functor": SmoothnessFunctor(self._params),
        "step_count_functor" : StepCountFunctor(self._params),
        "min_max_vel_functor" : MinMaxVelFunctor(self._params),
        "pot_center_functor": PotentialCenterlineFunctor(self._params),
        "pot_vel_functor": PotentialVelocityFunctor(self._params)
      })
    
class EvaluatorConfigurator(GeneralEvaluator):
  def __init__(self, params):
    self._params = params

    try:
      quantized = self._params["ML"]["EvaluatorConfigurator"]["RulesConfigs"]["quantized"]
    except KeyError:
      quantized = False

    # rule_functor_prefix = "TrafficRuleSTL" if quantized else "TrafficRuleLTL"
    # rule_functor_name = f"{rule_functor_prefix}Functor"

    # rule_impl_prefix = "traffic_rule_stl" if quantized else "traffic_rule_ltl"
    # rule_impl_name = f"{rule_impl_prefix}_functor"

    # add mapping of functors to keys
    self._fn_key_map = {
      "CollisionFunctor" : "collision_functor",
      "GoalFunctor" : "goal_functor",
      "LowSpeedGoalFunctor" : "low_speed_goal_reached_functor",
      "DrivableAreaFunctor" : "drivable_area_functor",
      "StepCountFunctor" : "step_count_functor",
      "SmoothnessFunctor" : "smoothness_functor",
      "MinMaxVelFunctor" : "min_max_vel_functor",
      "PotentialCenterlineFunctor": "pot_center_functor",
      "PotentialVelocityFunctor": "pot_vel_functor",
      "PotentialGoalSwitchVelocityFunctor": "pot_goal_switch_vel_functor",
      "PotentialGoalCenterlineFunctor": "pot_goal_center_functor",
      "StateActionLoggingFunctor": "state_action_logging_functor",
      "CollisionDrivableAreaFunctor" : "collision_drivable_area_functor",
      "PotentialGoalReachedVelocityFunctor": "pot_goal_vel_functor",
      "MaxStepCountAsGoalFunctor": "max_step_count_as_goal_functor",
      "PotentialGoalPolyFunctor": "pot_goal_poly_functor",      
      # rule_functor_name: rule_impl_name
    }
    
    functor_configs = self._params["ML"]["EvaluatorConfigurator"]["EvaluatorConfigs"]["FunctorConfigs"]
    functor_config_params_dict = functor_configs.ConvertToDict()

    # initialize functor and functorweights dicts
    eval_fns = {}
    # get values for each item
    for key in functor_config_params_dict.keys():
      matched_functor_key = self._fn_key_map[key]
      eval_fns[matched_functor_key]= eval("{}(functor_configs)".format(key))
    
    # initialize evaluators for bark world
    bark_evals = {
      "goal_reached" : lambda: EvaluatorGoalReached(),
      "collision" : lambda: EvaluatorCollisionEgoAgent(),
      "step_count" : lambda: EvaluatorStepCount(),
      "drivable_area" : lambda: EvaluatorDrivableArea()
    }
    
    rules_configs = self._params["ML"]["EvaluatorConfigurator"]["RulesConfigs"]

    for rule_config in rules_configs["Rules"]:
      # print("Rule name:", rule_config["RuleName"])

      # parse label function for each rule
      labels_list = []

      for label_conf in rule_config["RuleConfig"]["labels"]:
        label_params_dict = label_conf["params"].ConvertToDict()

        if label_conf["type"] == "EgoBeyondPointLabelFunction" or label_conf["type"] == "AgentBeyondPointLabelFunction":
          merge_point = label_params_dict["point"]
          label_params_dict["point"] = Point2d(merge_point[0],merge_point[1])
        label = eval("{}(*(label_params_dict.values()))".format(label_conf["type"]))  
        labels_list.append(label)

      # instance rule evaluator for each rule
      #TODO: check if evaluatorLTL can access private function in python
      tl_formula_ = rule_config["RuleConfig"]["params"]["formula"]
      # print("ltl_formula_:",tl_formula_)
      # print("labels_list:",labels_list)

      try:
        eval_return_robustness_only = rule_config["RuleConfig"]["params"]["eval_return_robustness_only"]        
      except KeyError:
        eval_return_robustness_only = True

      tmp_tl_settings = {}
      # Check if the key exists in tmp_tl_settings; if not, create a nested dictionary
      if rule_config["RuleName"] not in tmp_tl_settings:
        tmp_tl_settings[rule_config["RuleName"]] = {}
        
      tmp_tl_settings[rule_config["RuleName"]]["agent_id"] = 1
      tmp_tl_settings[rule_config["RuleName"]]["ltl_formula"] = tl_formula_
      tmp_tl_settings[rule_config["RuleName"]]["label_functions"] = labels_list

      if quantized:
        tmp_tl_settings[rule_config["RuleName"]]["eval_return_robustness_only"] = eval_return_robustness_only

      tmp_tl_eval = eval("{}(**tmp_tl_settings[rule_config['RuleName']])".format(rule_config["RuleConfig"]["type"]))
      # print("tmp_tl_eval:", tmp_tl_eval)
      # print("lambda: tmp_tl_eval: ", lambda: tmp_tl_eval)
      bark_evals[rule_config["RuleName"]] = tmp_tl_eval
      # bark_evals[rule_config["RuleName"]] = lambda: tmp_tl_eval
      
      # add rule functors to bark_ml_eval_fns
      functor_n_ = rule_config["RuleName"] + "_stl_functor" if quantized else "_ltl_functor"
      eval_fns[functor_n_] = eval("{}(rule_config)".format("TrafficRuleSTLFunctor" if quantized else "TrafficRuleLTLFunctor"))

    # print("bark_evals: ", bark_evals)
    # print("eval_fns: ", eval_fns)

    super().__init__(params=self._params, bark_eval_fns=bark_evals, bark_ml_eval_fns=eval_fns)
