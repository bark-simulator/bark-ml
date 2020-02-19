import numpy as np
from modules.runtime.runtime import Runtime

class RuntimeRL(Runtime):
  """Runtime wrapper for reinforcement learning.
     Extends the runtime with observers and evaluators.
  
  Arguments:
      Runtime {Runtime} -- BaseClass
  """
  def __init__(self,
               action_wrapper,
               observer,
               evaluator,
               step_time,
               viewer,
               scenario_generator=None,
               render=False):
    Runtime.__init__(self,
                     step_time=step_time,
                     viewer=viewer,
                     scenario_generator=scenario_generator,
                     render=render)
    self._action_wrapper = action_wrapper
    self._observer = observer
    self._evaluator = evaluator

  def reset(self, scenario=None):
    """Resets the runtime and its objects
    """
    super().reset(scenario=scenario)
    self._world = self._observer.reset(self._world,
                                       self._scenario._eval_agent_ids)
    self._world = self._evaluator.reset(self._world,
                                        self._scenario._eval_agent_ids)
    self._world = self._action_wrapper.reset(self._world,
                                             self._scenario._eval_agent_ids)
    # TODO(@hart): could be multiple
    observed_world = self._world.Observe(
      self._scenario._eval_agent_ids)[0]
    return self._observer.observe(observed_world)

  def step(self, action):
    """Steps the world with a specified time dt
    
    Arguments:
        action {any} -- will be passed to the ActionWrapper
    
    Returns:
        (next_state, reward, done, info) -- RL tuple
    """
    # TODO(@hart): could be multiple actions
    self._world = self._action_wrapper.action_to_behavior(world=self._world,
                                                          action=action)
    
    self._world.Step(self._step_time)
    snapshot =  self.snapshot(
      world=self._world,
      controlled_agents=self._scenario._eval_agent_ids,
      action=action)

    if self._render:
      self.render()
    return snapshot

  @property
  def action_space(self):
    """Action space of the agent
    """
    return self._action_wrapper.action_space

  @property
  def observation_space(self):
    """Observation space of the agent
    """
    return self._observer.observation_space

  def snapshot(self, world, controlled_agents, action):
    """Evaluates and observes the world from the controlled-agents's
       perspective
    
    Arguments:
        world {bark.world} -- Contains all objects of the simulation
        controlled_agents {list} -- list of AgentIds
    
    Returns:
        (next_state, reward, done, info) -- RL tuple
    """
    # TODO(@hart): could be multiple
    observed_world = self._world.Observe(controlled_agents)[0]
    next_state = self._observer.observe(observed_world)
    reward, done, info = self._evaluator.evaluate(
      world=world,
      action=action,
      observed_state=next_state)
    return next_state, reward, done, info


