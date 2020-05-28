import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# BARK imports
#from bark_project.modules.runtime.commons.parameters import ParameterServer

# tf2rl imports
import tf2rl

# BARK-ML imports


class TF2RLRunner:
  """Base class for runners based on tf2rl library"""

  def __init__(self,
                environment=None,
                agent=None,
                params=None):
      
    self._params = params
    self._agent = agent
    self._environment = environment

    self._trainer = self._GetTrainer()


  def GetTrainer(self):
    """Creates an trainer instance. Agent specific."""
    raise NotImplementedError

  def Train(self):
    """trains the agent."""
    self._train()


  def _train(self):
    """agent specific training method."""
    raise NotImplementedError


  def Evaluate(self):
    """Evaluates the agent."""

    # TODO: use the logger of the trainer object.
    #logger.info("Evaluating the agent's performance in {} episodes."
    #.format(str(self._params["ML"]["Runner"]["evaluation_steps"])))

    rewards = []
    steps = []

    for _ in range(0, self._params["ML"]["Runner"]["evaluation_steps"]):
      obs = self._environment.reset()
      is_terminal = False

      while not is_terminal:
        action = self._agent.Act(obs)
        obs, reward, is_terminal, _ = self._environment.step(action)
        rewards.append(reward)
        steps.append(1)

    mean_reward = np.sum(np.array(rewards)) / self._params["ML"]["Runner"]["evaluation_steps"]
    mean_steps = np.sum(np.array(steps)) / self._params["ML"]["Runner"]["evaluation_steps"]

    # TODO: use the summary writer of the trainer
    #tf.summary.scalar("mean_reward",
    #                mean_reward)
    #tf.summary.scalar("mean_steps",
    #                mean_steps)

    #logger.info(
    #"The agent achieved on average {} reward and {} steps in \
    #{} episodes." \
    #.format(str(mean_reward),
    #        str(mean_steps),
    #        str(self._params["ML"]["Runner"]["evaluation_steps"])))

    

  def Visualize(self, num_episodes=1):
    """Visualizes the agent."""

    for _ in range(0, num_episodes):
      obs = self._environment.reset()
      is_terminal = False

      while not is_terminal:
        print(obs)
        action = self._agent.Act(obs)
        obs, reward, is_terminal, _ = self._environment.step(action)
        print(reward)
        self._environment.render()

    
  def SetupSummaryWriter(self):
    """Not sure whether necessary or not"""

    """if self._params["ML"]["TFARunner"]["SummaryPath"] is not None:
        try:
        self._summary_writer = tf.summary.create_file_writer(
            self._params["ML"]["TFARunner"]["SummaryPath"])
        except:
        pass
    self.get_initial_collection_driver()
    self.get_collection_driver()
    """

  def GetInitialCollectionDriver(self):
    """Not sure whether necessary or not"""

    """self._initial_collection_driver = \
        dynamic_episode_driver.DynamicEpisodeDriver(
        env=self._wrapped_env,
        policy=self._agent._agent.collect_policy,
        observers=[self._agent._replay_buffer.add_batch],
        num_episodes=self._params["ML"]["TFARunner"]["InitialCollectionEpisodes", "", 50])
        """

  def GetCollectionDriver(self):
    """Not sure whether necessary or not"""

    """self._collection_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        env=self._wrapped_env,
        policy=self._agent._agent.collect_policy,
        observers=[self._agent._replay_buffer.add_batch],
        num_episodes=self._params["ML"]["TFARunner"]["CollectionEpisodesPerStep", "", 1])
      """

  def CollectInitialEpisodes(self):
    """Not sure whether necessary or not"""

    """self._initial_collection_driver.run()"""

  
