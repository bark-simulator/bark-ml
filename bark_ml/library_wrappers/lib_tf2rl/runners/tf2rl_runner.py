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

        # TODO set up logger
        """self._eval_metrics = [
        tf_metrics.AverageReturnMetric(
            buffer_size=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 25]),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 25])
        ]
        
        self._summary_writer = None
        self._params = params or ParameterServer()
        
        self._wrapped_env = tf_py_environment.TFPyEnvironment(
        TFAWrapper(self._environment))
        self.GetInitialCollectionDriver()
        self.GetCollectionDriver()
        self._logger = logging.getLogger()"""


    def GetTrainer(self):
    """Creates an trainer instance.
    Should be written in the trainer class specified for an agent type.
    """
    pass

    def Train(self):
        """trains the agent."""
        self._train()


    def Evaluate(self):
        """Evaluates the agent."""
        ##################################################################
        #global_iteration = self._agent._agent._train_step_counter.numpy()
        ##################################################################

        # TODO : set up logger
        """logger.info("Evaluating the agent's performance in {} episodes."
        .format(str(self._params["ML"]["Runner"]["evaluation_steps"])))"""

        rewards = []
        steps = []
        for _ in range(0, self._params["ML"]["TF2RLRunner"]["evaluation_steps"]):
            obs = self._unwrapped_runtime.reset()
            is_terminal = False

            while not is_terminal:
                action = self._agent.action(obs)
                obs, reward, is_terminal, _ = self._environment.step(action)
                rewards.append(reward)
                steps.append(1)

        mean_reward = np.sum(np.array(rewards))/self._params["ML"]["TF2RLRunner"]["evaluation_steps"]
        mean_steps = np.sum(np.array(steps))/self._params["ML"]["TF2RLRunner"]["evaluation_steps"]
        tf.summary.scalar("mean_reward",
                        mean_reward)
        tf.summary.scalar("mean_steps",
                        mean_steps)

        #########################################
        #tf.summary.scalar("mean_reward",
        #                  mean_reward,
        #                  step=global_iteration)
        #tf.summary.scalar("mean_steps",
        #                  mean_steps,
        #                  step=global_iteration)
        #########################################
        """logger.info(
        "The agent achieved on average {} reward and {} steps in \
        {} episodes." \
        .format(str(mean_reward),
                str(mean_steps),
                str(self._params["ML"]["Runner"]["evaluation_steps"])))"""


    def Visualize(self):
        """Visualizes the agent."""


    def SetupSummaryWriter(self):
    if self._params["ML"]["TFARunner"]["SummaryPath"] is not None:
        try:
        self._summary_writer = tf.summary.create_file_writer(
            self._params["ML"]["TFARunner"]["SummaryPath"])
        except:
        pass
    self.get_initial_collection_driver()
    self.get_collection_driver()

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
    """Agent specific
    """
    pass

    def Evaluate(self):
    self._agent._training = False
    global_iteration = self._agent._agent._train_step_counter.numpy()
    self._logger.info("Evaluating the agent's performance in {} episodes."
        .format(str(self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20])))
    metric_utils.eager_compute(
        self._eval_metrics,
        self._wrapped_env,
        self._agent._agent.policy,
        num_episodes=self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20])
    metric_utils.log_metrics(self._eval_metrics)
    tf.summary.scalar("mean_reward",
                        self._eval_metrics[0].result().numpy(),
                        step=global_iteration)
    tf.summary.scalar("mean_steps",
                        self._eval_metrics[1].result().numpy(),
                        step=global_iteration)
    self._logger.info(
        "The agent achieved on average {} reward and {} steps in \
        {} episodes." \
        .format(str(self._eval_metrics[0].result().numpy()),
                str(self._eval_metrics[1].result().numpy()),
                str(self._params["ML"]["TFARunner"]["EvaluationSteps", "", 20])))


    def Visualize(self, num_episodes=1):
    self._agent._training = False
    for _ in range(0, num_episodes):
        state = self._environment.reset()
        is_terminal = False
        while not is_terminal:
        action_step = self._agent._eval_policy.action(ts.transition(state, reward=0.0, discount=1.0))
        state, reward, is_terminal, _ = self._environment.step(action_step.action.numpy())
        self._environment.render()
