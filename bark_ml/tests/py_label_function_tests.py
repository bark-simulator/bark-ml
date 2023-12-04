# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle, and
# Tobias Kessler
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import unittest
import time
import math
from math import pi

from bark.runtime.commons.parameters import ParameterServer
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint, \
  ContinuousSingleLaneBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.evaluators.evaluator_configs import GoalReached,RewardShapingEvaluator,EvaluatorConfigurator
from bark_ml.evaluators.general_evaluator import GeneralEvaluator
from bark_ml.evaluators.stl.evaluator_stl import EvaluatorSTL
from bark_ml.evaluators.stl.label_functions.safe_distance_label_function import SafeDistanceQuantizedLabelFunction
from bark.core.world import make_test_world
from bark.core.world.goal_definition import GoalDefinitionPolygon

class PyLabelFunctionTests(unittest.TestCase):

  def test_safe_distance_longitudinal(self):
    ego_id = 1
    v_0 = 8.0
    dv = 0.0
    delta = 1.0
    a_e = -8.0
    a_o = -8.0

    # Create a SafeDistanceQuantizedLabelFunction instance
    evaluator = SafeDistanceQuantizedLabelFunction("safe_distance", False, delta, delta, a_e, a_o, True, 4, False, 1.0, math.pi, False)

    stop_dist = v_0 * delta + (v_0 ** 2) / (2.0 * -a_e)

    # Case 1
    assert stop_dist > 1.0

    # Create a test world 
    world = make_test_world(1, stop_dist - 1.0, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)

    # Observe the world and get the observed world for the ego agent
    observed_world = world.Observe([ego_id])[0]
    
    # Evaluate label function
    labels = evaluator.Evaluate(observed_world)

    self.assertTrue(next(iter(labels.values())))    
    self.assertTrue(evaluator.robustness >= 0.0)
    print(f"test_safe_distance_longitudinal - Case 1 passed (result: {next(iter(labels.values()))}, robustness={evaluator.robustness})")

    # Case 2
    dist = 5.0
    world2 = make_test_world(1, dist, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)
    observed_world2 = world2.Observe([ego_id])[0]
    labels2 = evaluator.Evaluate(observed_world2)
    self.assertFalse(next(iter(labels2.values())))    
    self.assertTrue(evaluator.robustness <= 0.0)
    print(f"test_safe_distance_longitudinal - Case 2 passed (result: {next(iter(labels2.values()))}, robustness={evaluator.robustness})")

    # Case 3
    dist = 2.0
    world3 = make_test_world(1, dist, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)
    observed_world3 = world3.Observe([ego_id])[0]
    labels3 = evaluator.Evaluate(observed_world3)
    self.assertFalse(next(iter(labels3.values())))
    self.assertTrue(evaluator.robustness <= 0.0)
    print(f"test_safe_distance_longitudinal - Case 3 passed (result: {next(iter(labels3.values()))}, robustness={evaluator.robustness})")

    # Case 4
    delta = 0.5
    dist = 4.5
    evaluator = SafeDistanceQuantizedLabelFunction("safe_distance", False, delta, delta, a_e, a_o, True, 4, False, 1.0, math.pi, False)
    world4 = make_test_world(1, dist, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)
    observed_world4 = world4.Observe([ego_id])[0]
    labels4 = evaluator.Evaluate(observed_world4)
    self.assertTrue(next(iter(labels4.values())))
    self.assertTrue(evaluator.robustness >= 0.0)
    print(f"test_safe_distance_longitudinal - Case 4 passed (result: {next(iter(labels4.values()))}, robustness={evaluator.robustness})")

    # Case 5
    dist = 6.0
    world5 = make_test_world(1, dist, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)
    observed_world5 = world5.Observe([ego_id])[0]
    labels5 = evaluator.Evaluate(observed_world5)
    self.assertTrue(next(iter(labels5.values())))
    self.assertTrue(evaluator.robustness >= 0.0)
    print(f"test_safe_distance_longitudinal - Case 5 passed (result: {next(iter(labels5.values()))}, robustness={evaluator.robustness})")

  def test_safe_distance_lateral(self):
      v_0 = 8.0
      dv = 0.0
      delta = 1.0
      a_e = -8.0
      a_o = -8.0

      evaluator = SafeDistanceQuantizedLabelFunction("safe_distance", False, delta, delta, a_e, a_o, True, 4, False, 5.0, math.pi, True)

      # Longitudinal safe dist not violated -> lateral on same lane -> no violation
      stop_dist = v_0 * delta + v_0 * v_0 / (2.0 * -a_e)
      assert stop_dist > 1.0
      dist_lat = 0.0
      angle = 0.0

      world = make_test_world(1, stop_dist - 1.0, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world.GetAgents().values())).id
      observed_world = world.Observe([ego_agent_id])[0]
      labels = evaluator.Evaluate(observed_world)

      self.assertTrue(next(iter(labels.values())))
      self.assertTrue(evaluator.robustness >= 0.0)
      print(f"test_safe_distance_lateral - Case 1 passed (result: {next(iter(labels.values()))}, robustness={evaluator.robustness})")

      # Longitudinal safe dist violated -> lateral on same lane -> violated
      dist_long = 5.0
      dist_lat = 0.0
      angle = 0.0

      world2 = make_test_world(1, dist_long, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world2.GetAgents().values())).id
      observed_world2 = world2.Observe([ego_agent_id])[0]
      labels2 = evaluator.Evaluate(observed_world2)
      self.assertFalse(next(iter(labels2.values())))
      self.assertTrue(evaluator.robustness <= 0.0)
      print(f"test_safe_distance_lateral - Case 2 passed (result: {next(iter(labels2.values()))}, robustness={evaluator.robustness})")

      # Longitudinal safe dist violated -> lateral on right of ego zero lat velocity -> not violated
      dist_long = 0.0
      dist_lat = 2.5
      angle = 0.0
  
      world3 = make_test_world(1, dist_long, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world3.GetAgents().values())).id
      observed_world3 = world3.Observe([ego_agent_id])[0]
      labels3 = evaluator.Evaluate(observed_world3)
      self.assertTrue(next(iter(labels3.values())))
      self.assertTrue(evaluator.robustness >= 0.0)
      print(f"test_safe_distance_lateral - Case 3 passed (result: {next(iter(labels3.values()))}, robustness={evaluator.robustness})")

      # Longitudinal safe dist violated -> lateral on left of ego, lat velocity away from ego -> not violated 
      dist_long = 3.0
      angle = math.pi / 4.0
      dist_lat = 3.5
      
      world4 = make_test_world(1, dist_long, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world4.GetAgents().values())).id
      observed_world4 = world4.Observe([ego_agent_id])[0]
      labels4 = evaluator.Evaluate(observed_world4)
      self.assertTrue(next(iter(labels4.values())))
      self.assertTrue(evaluator.robustness >= 0.0)
      print(f"test_safe_distance_lateral - Case 4 passed (result: {next(iter(labels4.values()))}, robustness={evaluator.robustness})")  

      # Longitudinal safe dist violated -> lateral on left of ego, lat velocity towards ego -> violated 
      dist_long = 3.0
      angle = -math.pi / 4.0
      dist_lat = 3.5
      world5 = make_test_world(1, dist_long, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world5.GetAgents().values())).id
      observed_world5 = world5.Observe([ego_agent_id])[0]
      labels5 = evaluator.Evaluate(observed_world5)
      self.assertFalse(next(iter(labels5.values())))
      self.assertTrue(evaluator.robustness <= 0.0)
      print(f"test_safe_distance_lateral - Case 5 passed (result: {next(iter(labels5.values()))}, robustness={evaluator.robustness})")

  def test_evaluator_stl(self):        
      ego_id = 1
      v_0 = 8.0
      dv = 0.0
      delta = 1.0
      a_e = -8.0
      a_o = -8.0

      # Create a SafeDistanceQuantizedLabelFunction instance
      sdf = SafeDistanceQuantizedLabelFunction("safe_distance", False, delta, delta, a_e, a_o, True, 4, False, 1.0, math.pi, False)

      stop_dist = v_0 * delta + (v_0 ** 2) / (2.0 * -a_e)

      # Case 1
      assert stop_dist > 1.0

      # Create a test world (you need to implement the make_test_world function)
      world = make_test_world(1, stop_dist - 1.0, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)

      # Observe the world and get the observed world for the ego agent
      observed_world = world.Observe([ego_id])[0]
      
      # Evaluate labels using the evaluator
      evaluator_stl = EvaluatorSTL(1, "G safe_distance", [sdf])

      start_time = time.time()
      evaluator_stl.Evaluate(observed_world) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lon Case 1 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

      # Case 2
      dist = 5.0
      world2 = make_test_world(1, dist, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)
      observed_world2 = world2.Observe([ego_id])[0]
      
      start_time = time.time()
      evaluator_stl.Evaluate(observed_world2) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lon Case 2 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

      # Case 3
      dist = 2.0
      world3 = make_test_world(1, dist, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)
      observed_world3 = world3.Observe([ego_id])[0]
      
      start_time = time.time()
      evaluator_stl.Evaluate(observed_world3) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lon Case 3 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

      # Case 4
      delta = 0.5
      dist = 4.5
      sdf = SafeDistanceQuantizedLabelFunction("safe_distance", False, delta, delta, a_e, a_o, True, 4, False, 1.0, math.pi, False)
      world4 = make_test_world(1, dist, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)
      observed_world4 = world4.Observe([ego_id])[0]
      evaluator_stl = EvaluatorSTL(1, "G safe_distance", [sdf])

      start_time = time.time()
      evaluator_stl.Evaluate(observed_world4) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lon Case 4 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

      # Case 5
      dist = 6.0
      world5 = make_test_world(1, dist, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, 0.0, 0.0)
      observed_world5 = world5.Observe([ego_id])[0]
      
      start_time = time.time()
      evaluator_stl.Evaluate(observed_world5) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lon Case 5 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

      ego_id = 1 
      v_0 = 8.0
      dv = 0.0
      delta = 1.0
      a_e = -8.0
      a_o = -8.0

      sdf = SafeDistanceQuantizedLabelFunction("safe_distance", False, delta, delta, a_e, a_o, True, 4, False, 5.0, math.pi, True)

      # Longitudinal safe dist not violated -> lateral on same lane -> no violation
      stop_dist = v_0 * delta + v_0 * v_0 / (2.0 * -a_e)
      assert stop_dist > 1.0
      dist_lat = 0.0
      angle = 0.0

      world = make_test_world(1, stop_dist - 1.0, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world.GetAgents().values())).id
      observed_world = world.Observe([ego_agent_id])[0]
      evaluator_stl = EvaluatorSTL(1, "G safe_distance", [sdf])

      start_time = time.time()
      evaluator_stl.Evaluate(observed_world) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lat Case 1 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

      # Longitudinal safe dist violated -> lateral on same lane -> violated
      dist_long = 5.0
      dist_lat = 0.0
      angle = 0.0

      world2 = make_test_world(1, dist_long, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world2.GetAgents().values())).id
      observed_world2 = world2.Observe([ego_agent_id])[0]
      
      start_time = time.time()
      evaluator_stl.Evaluate(observed_world2) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lat Case 2 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

      # Longitudinal safe dist violated -> lateral on right of ego zero lat velocity -> not violated
      dist_long = 0.0
      dist_lat = 2.5
      angle = 0.0
  
      world3 = make_test_world(1, dist_long, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world3.GetAgents().values())).id
      observed_world3 = world3.Observe([ego_agent_id])[0]
      
      start_time = time.time()
      evaluator_stl.Evaluate(observed_world3) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lat Case 3 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

      # Longitudinal safe dist violated -> lateral on left of ego, lat velocity away from ego -> not violated 
      dist_long = 3.0
      angle = math.pi / 4.0
      dist_lat = 3.5

      world4 = make_test_world(1, dist_long, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world4.GetAgents().values())).id
      
      start_time = time.time()
      evaluator_stl.Evaluate(observed_world4) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lat Case 4 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

      # Longitudinal safe dist violated -> lateral on left of ego, lat velocity towards ego -> violated 
      dist_long = 3.0
      angle = -math.pi / 4.0
      dist_lat = 3.5
      
      world5 = make_test_world(1, dist_long, v_0, dv, GoalDefinitionPolygon(), 0.0, 0.0, dist_lat, angle)
      ego_agent_id = next(iter(world5.GetAgents().values())).id
      
      start_time = time.time()
      evaluator_stl.Evaluate(observed_world5) 
      end_time = time.time()
      self.assertTrue(evaluator_stl.robustness == sdf.robustness)    
      print(f"test_evaluator_stl - Lat Case 5 passed (robustness={evaluator_stl.robustness}, elapsed time: {end_time-start_time} seconds.)")

if __name__ == '__main__':
  unittest.main()
