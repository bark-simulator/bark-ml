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

        # TODO 
        # set up logger
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
        # TODO 
        # write Evaluate method which is general for all tf2rl agents
        # this was just copied from the runner file for the previous version bark-ml

        """
        ##################################################################
        #global_iteration = self._agent._agent._train_step_counter.numpy()
        ##################################################################
        logger.info("Evaluating the agent's performance in {} episodes."
        .format(str(self._params["ML"]["Runner"]["evaluation_steps"])))
        # Ticket (https://github.com/tensorflow/agents/issues/59) recommends
        # to do the rendering in the original environment
        rewards = []
        steps = []
        if self._unwrapped_runtime is not None:
        for _ in range(0, self._params["ML"]["Runner"]["evaluation_steps"]):
            obs = self._unwrapped_runtime.reset()
            is_terminal = False

            while not is_terminal:
            action = self._agent._generator.get_action(obs)
            obs, reward, is_terminal, _ = self._unwrapped_runtime.step(action)
            rewards.append(reward)
            steps.append(1)

        mean_reward = np.sum(np.array(rewards))/self._params["ML"]["Runner"]["evaluation_steps"]
        mean_steps = np.sum(np.array(steps))/self._params["ML"]["Runner"]["evaluation_steps"]
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
        logger.info(
        "The agent achieved on average {} reward and {} steps in \
        {} episodes." \
        .format(str(mean_reward),
                str(mean_steps),
                str(self._params["ML"]["Runner"]["evaluation_steps"])))

        """

    def Visualize(self):
        """Visualizes the agent."""
        # TODO 
        # write Visualize method which is general for all tf2rl agents
        # this was just copied from the runner file for the previous version bark-ml

        """
        if self._unwrapped_runtime is not None:
            for _ in range(0, num_episodes):
            obs = self._unwrapped_runtime.reset()
            is_terminal = False
            while not is_terminal:
                print(obs)
                action = self._agent._generator.get_action(obs)
                # TODO(@hart); make generic for multi agent planning
                obs, reward, is_terminal, _ = self._unwrapped_runtime.step(action)
                print(reward)
                self._unwrapped_runtime.render()

        """


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

    
