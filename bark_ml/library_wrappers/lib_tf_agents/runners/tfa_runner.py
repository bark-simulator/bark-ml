# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import logging
import tensorflow as tf
import numpy as np
tf.compat.v1.enable_v2_behavior()

# BARK imports
from bark.runtime.commons.parameters import ParameterServer
from bark.runtime.viewer.buffered_viewer import BufferedViewer
from bark.core.geometry import *
from bark.core.world.renderer import *

# tf agent imports
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.trajectories import time_step as ts

# BARK-ML imports
from bark_ml.library_wrappers.lib_tf_agents.py_bark_environment import PyBARKEnvironment
from bark_ml.commons.tracer import Tracer


class TFARunner:
  """Used to train, evaluate and visualize a BARK-ML agent."""

  def __init__(self,
               environment=None,
               agent=None,
               tracer=None,
               params=None):
    self._params = params or ParameterServer()
    self._eval_metrics = [
      tf_metrics.AverageReturnMetric(
        buffer_size=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 25]),
      tf_metrics.AverageEpisodeLengthMetric(
        buffer_size=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 25])
    ]
    self._agent = agent
    self._agent.set_action_externally = True
    self._summary_writer = None
    self._environment = environment
    self._wrapped_env = tf_py_environment.TFPyEnvironment(
      PyBARKEnvironment(self._environment))
    self.GetInitialCollectionDriver()
    self.GetCollectionDriver()
    self._logger = logging.getLogger()
    self._tracer = tracer or Tracer()
    self._colliding_scenario_ids = []

  def SetupSummaryWriter(self):
    if self._params["ML"]["TFARunner"]["SummaryPath"] is not None:
      try:
        self._summary_writer = tf.summary.create_file_writer(
          self._params["ML"]["TFARunner"]["SummaryPath"])
      except:
        pass
    self.GetInitialCollectionDriver()
    self.GetCollectionDriver()

  def GetInitialCollectionDriver(self):
    self._initial_collection_driver = \
      dynamic_episode_driver.DynamicEpisodeDriver(
        env=self._wrapped_env,
        policy=self._agent._agent.collect_policy,
        observers=[self._agent._replay_buffer.add_batch],
        num_episodes=self._params["ML"]["TFARunner"]["InitialCollectionEpisodes", "", 50])

  def GetCollectionDriver(self):
    self._collection_driver = dynamic_episode_driver.DynamicEpisodeDriver(
      env=self._wrapped_env,
      policy=self._agent._agent.collect_policy,
      observers=[self._agent._replay_buffer.add_batch],
      num_episodes=self._params["ML"]["TFARunner"]["CollectionEpisodesPerStep", "", 1])

  def CollectInitialEpisodes(self):
    self._initial_collection_driver.run()

  def Train(self):
    self.CollectInitialEpisodes()
    if self._summary_writer is not None:
      with self._summary_writer.as_default():
        self._train()
    else:
      self._train()

  def _train(self):
    """Agent specific."""
    pass

  def ReshapeActionIfRequired(self, action_step):
    action_shape = action_step.action.shape
    expected_shape = self._agent._eval_policy.action_spec.shape
    action = action_step.action.numpy()
    if action_shape != expected_shape:
      # logging.warning("Action shape" + str(action_shape) + \
      #   " does not match with expected shape " + str(expected_shape) +\
      #   " -> reshaping is tried")
      action = np.reshape(action, expected_shape)
      # logging.info(action)
    return action

  @staticmethod
  def _id_to_idx(id_agent_id_map, aid):
    return list(id_agent_id_map.keys())[list(id_agent_id_map.values()).index(aid)]

  @staticmethod
  def _id_agent_map(world, obs, ego_id):
    ego_agent = world.agents[ego_id]
    agents = list(world.agents.values())
    agents.remove(ego_agent)
    agents = obs._agents_sorted_by_distance(ego_agent, agents)
    agents.insert(0, ego_agent)
    agents = agents[:obs._num_agents]
    id_agent_id_map = {}
    for idx, agent in enumerate(agents):
      id_agent_id_map[idx] = agent.id
    return id_agent_id_map

  @staticmethod
  def _get_agent_pos(world, aid):
    agent = world.agents[aid]
    return [agent.state[1], agent.state[2]]

  def ProcessGraphTuple(self, env, graph_tuple, ego_id, render=False):
    senders = graph_tuple.senders.numpy()
    receivers = graph_tuple.receivers.numpy()
    edges = graph_tuple.edges.numpy()

    # sorted list
    id_agent_id_map = self._id_agent_map(env._world, env._observer, ego_id)
    # get idx
    for _, agent_id in id_agent_id_map.items():
      receiver_idx = np.where(
        receivers == self._id_to_idx(id_agent_id_map, ego_id))
      sender_idx = senders[receiver_idx]
      for sid in sender_idx:
        from_id = agent_id
        to_id = id_agent_id_map[sid]
        from_pos = self._get_agent_pos(env._world, from_id)
        to_pos = self._get_agent_pos(env._world, to_id)
        # TODO: use sum(..), sum(abs(..)), min(..), max(..), ...
        norm = np.sum(np.fabs(edges), axis=(1, 0))
        magnitude = np.sum(edges[sid])/norm
        color = (128/255, 128/255, 128/255, .5)
        zorder = 1
        alpha = 0.4
        if agent_id == ego_id:
          color = (12/255, 44/255, 132/255, 1.)
          zorder = 10
          alpha = 0.8
        from_agent_state = env._world.agents[from_id].state
        to_agent_state = env._world.agents[to_id].state
        if to_id != agent_id:
          self._tracer.Trace(
            {"length_dx": (from_pos[0] - to_pos[0]),
            "length_dy": (from_pos[1] - to_pos[1]),
            "dv": from_agent_state[4] - to_agent_state[4],
            # "dtheta": SignedAngleDiff(from_agent_state[3], to_agent_state[3]),
            "magnitude": np.sum(edges[sid])})
        if render:
          if isinstance(env._viewer, BufferedViewer):
            l = Line2d()
            l.AddPoint(Point2d(from_pos[0], from_pos[1]))
            l.AddPoint(Point2d(to_pos[0], to_pos[1]))
            line_primitive = RenderPrimitive(l)
            if agent_id == ego_id:
              line_primitive.Add("stroke_color", [12, 44, 132, 255])
            else:
              line_primitive.Add("stroke_color", [128, 128, 128, 128])
            line_primitive.Add("stroke_width", max(5*magnitude, .1))
            env._world.renderer.Add("LINES", line_primitive)
          else:
            ax = env._viewer.axes
            ax.plot(
              [from_pos[0], to_pos[0]],
              [from_pos[1], to_pos[1]],
              color=color, marker="o", linewidth=max(min(50*magnitude, 5.), 0.5), zorder=zorder, alpha=alpha)

    if render:
      for _, agent_id in id_agent_id_map.items():
        agent_pos = self._get_agent_pos(env._world, agent_id)
        color = "gray"
        if agent_id == ego_id:
          color = "blue"
        if isinstance(env._viewer, BufferedViewer):
          p = Point2d(agent_pos[0], agent_pos[1])
          point_primitive = RenderPrimitive(p)
          point_primitive.Add("stroke_color", [12, 44, 132, 255])
          point_primitive.Add("radius_pixels", 0.3)
          point_primitive.Add("fill_color", [12, 44, 132, 128])
          env._world.renderer.Add("POINTS", point_primitive)
        else:
          ax = env._viewer.axes
          ax.plot(
            agent_pos[0], agent_pos[1], marker='o', color=color, markersize=6)

  def RunEpisode(self, render=True, trace_colliding_ids=None, **kwargs):
    state = self._environment.reset()
    is_terminal = False
    # print(self._agent._eval_policy.trainable_variables)
    if render:
      self._environment.render()
    while not is_terminal:
      action_step = self._agent._eval_policy.action(
        ts.transition(state, reward=0.0, discount=1.0))
      action = self.ReshapeActionIfRequired(action_step)
      env_data = self._environment.step(action)
      if render:
        self._logger.info("Current state: {}.".format(["{0:1.2f}".format(i) for i in state]))
      self._tracer.Trace(env_data, **kwargs)
      state, reward, is_terminal, info = env_data
      # graph stuff
      # try:
      #   graph_tuples = self._agent._agent._actor_network._latent_trace
      #   ego_id = self._environment._scenario._eval_agent_ids[0]
      #   self.ProcessGraphTuple(
      #     self._environment,
      #     graph_tuples[-1],
      #     ego_id,
      #     render)
      # except:
      #   pass
      if render:
        self._logger.info(f"The ego agent's action is {action} and " + \
                          f"a reward of {reward}.")
        self._environment.render()
      if is_terminal and (info["collision"] or info["drivable_area"]) and trace_colliding_ids is not None:
        self._colliding_scenario_ids.append(
          self._environment._scenario_idx)
      if is_terminal and info["goal_reached"]:
        self._logger.info("\033[92mThe ego agent reached its goal. \033[0m")

  def Run(
    self, num_episodes=10, render=False, mode="not_training",
    trace_colliding_ids=None, **kwargs):
    for i in range(0, num_episodes):
      if render:
        self._logger.info(f"Simulating episode {i}.")
      self.RunEpisode(
        render=render, num_episode=i, trace_colliding_ids=trace_colliding_ids, **kwargs)

    mean_col_rate = self._tracer.collision_rate
    goal_reached = self._tracer.success_rate
    mean_reward = self._tracer.mean_reward
    mean_steps = self._tracer.mean_steps

    if mode == "training":
      global_iteration = self._agent._agent._train_step_counter.numpy()
      tf.summary.scalar("mean_reward", mean_reward, step=global_iteration)
      tf.summary.scalar("mean_steps", mean_steps, step=global_iteration)
      tf.summary.scalar(
        "mean_collision_rate", mean_col_rate, step=global_iteration)
      tf.summary.scalar(
        "goal_reached", goal_reached, step=global_iteration)

      res = {}
      for state in self._tracer._states:
        for key, val in state.items():
          if key not in res:
            res[key] = 0.
          res[key] += val

      for key, val in res.items():
        if key not in ["state", "goal_reached", "step_count", "num_episode", "reward"]:
          tf.summary.scalar(f"auto_{key}", val, step=global_iteration)

    print(
      f"The agent achieved an average reward of {mean_reward:.3f}," +
      f" collision-rate of {mean_col_rate:.5f}, took on average" +
      f" {mean_steps:.3f} steps, and reached the goal " +
      f" {goal_reached:.3f} (evaluated over {num_episodes} episodes).")

    if trace_colliding_ids:
      return self._colliding_scenario_ids
    else:
      return {
        "goal_reached": goal_reached,
        "mean_col_rate": mean_col_rate,
        "mean_steps": mean_steps,
        "mean_reward": mean_reward,
      }