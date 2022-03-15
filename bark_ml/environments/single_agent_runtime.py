# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from bark.runtime.runtime import Runtime


class SingleAgentRuntime(Runtime):
  """Single agent runtime where action is passed to the
  ego agent.

  Can either be initialized using a blueprint or by passing the
  `evaluator`, `observer`, `scenario_generation`, `step_time`, `viewer`
  and ml_behavior.
  """

  def __init__(self,
               blueprint=None,
               ml_behavior=None,
               observer=None,
               evaluator=None,
               step_time=None,
               viewer=None,
               art_part=None,
               num_scenarios=None,
               scenario_generator=None,
               odd_scenario_generator=None,
               render=False):

    if blueprint is not None:
      self._scenario_generator = blueprint._scenario_generation
      self._viewer = blueprint._viewer
      self._ml_behavior = blueprint._ml_behavior
      self._step_time = blueprint._dt
      self._evaluator = blueprint._evaluator
      self._observer = blueprint._observer
    Runtime.__init__(self,
                     step_time=step_time or self._step_time,
                     viewer=viewer or self._viewer,
                     scenario_generator = scenario_generator or blueprint._scenario_generation,
                     render=render)
    self._odd_scenario_generator = odd_scenario_generator
    self._ml_behavior = ml_behavior or self._ml_behavior
    self._observer = observer or self._observer
    self._evaluator = evaluator or self._evaluator
    self._world = None
    self._scenario_idx = -1
    self._art_part = 10
    # 80 percent of scenarios are artificial
    if art_part:
      self._art_part = int(art_part*10)
    self._num_scenarios = num_scenarios

  def reset(self, scenario=None):
    """Resets the runtime and its objects."""
    if scenario:
      self._scenario = scenario
    else:
      # pure odd scenarios
      if (self._scenario_generator is None) and (self._odd_scenario_generator is not None):
        self._scenario, self._scenario_idx = \
          self._odd_scenario_generator.get_next_scenario()
      # pure artificial scenario
      if (self._odd_scenario_generator is None) and (self._scenario_generator is not None):
        self._scenario, self._scenario_idx = \
          self._scenario_generator.get_next_scenario()
      # mixed odd and artificial scenario
      if (self._odd_scenario_generator is not None) and (self._scenario_generator is not None):
        cur_scen_idx = self._scenario_idx+1
        if cur_scen_idx >= self._num_scenarios:
          cur_scen_idx=0
        if cur_scen_idx % 10 < self._art_part:
          self._scenario, _ = \
            self._scenario_generator.get_next_scenario()
        else:
          self._scenario, _ = \
            self._odd_scenario_generator.get_next_scenario()

        self._scenario_idx = cur_scen_idx

    self._world = self._scenario.GetWorldState()
    self._reset_has_been_called = True
    if self._viewer:
      self._viewer.Reset()
    self._world_history = []
    assert len(self._scenario._eval_agent_ids) == 1, \
      "This runtime only supports a single agent!"
    eval_id = self._scenario._eval_agent_ids[0]
    self._world.UpdateAgentRTree()
    self._world = self._observer.Reset(self._world)
    self._world = self._evaluator.Reset(self._world)
    self._world.agents[eval_id].behavior_model = self._ml_behavior

    # render
    if self._render:
      self.render()

    # observe
    observed_world = self._world.Observe([eval_id])[0]
    return self._observer.Observe(observed_world)

  def step(self, action):
    # set actions
    eval_id = self._scenario._eval_agent_ids[0]
    if eval_id in self._world.agents:
      self._world.agents[eval_id].behavior_model.ActionToBehavior(action)

    # step and observe
    self._world.Step(self._step_time)
    observed_world = self._world.Observe([eval_id])

    if len(observed_world) > 0:
      observed_world = observed_world[0]
    else:
      raise Exception('No world instance available.')

    # observe and evaluate
    observed_next_state = self._observer.Observe(observed_world)
    reward, done, info = self._evaluator.Evaluate(
      observed_world=observed_world,
      action=action)

    # render
    if self._render:
      self.render()

    return observed_next_state, reward, done, info

  @property
  def action_space(self):
    """Action space of the agent."""
    return self._ml_behavior.action_space

  @property
  def observation_space(self):
    """Observation space of the agent."""
    return self._observer.observation_space

  @property
  def ml_behavior(self):
    return self._ml_behavior

  @ml_behavior.setter
  def ml_behavior(self, ml_behavior):
    self._ml_behavior = ml_behavior

