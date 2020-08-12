import gym
from bark_project.bark.runtime.commons.parameters import ParameterServer
from bark_project.bark.runtime.runtime import Runtime
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML
from bark_ml.library_wrappers.lib_tf2rl.agents.tf2rl_agent import BehaviorTF2RLAgent
from tf2rl.algos.gail import GAIL
from tf2rl.algos.ddpg import DDPG
import tf2rl
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import greedy_policy
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()


class BehaviorGAILAgent(BehaviorTF2RLAgent, BehaviorContinuousML):
    """GAIL agent based on the tf2rl library."""

    def __init__(self,
                 environment=None,
                 params=None):
        """constructor

        Args:
            environment (Runtime, optional): A environment with a gym
                environment interface. Defaults to None. 
            params (ParameterServer, optional): The parameter server
                 holding the settings. Defaults to None.
        """
        BehaviorTF2RLAgent.__init__(self,
                                    environment=environment,
                                    params=params)
        BehaviorContinuousML.__init__(self, params)

        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()

    def _get_generator(self):
        """Returns instantiated policy -
        parameters from ./examples/example_params/gail_params.json
        """
        generator_params = self._params["ML"]["BehaviorGAILAgent"]["Generator"]

        policy = DDPG(
            state_shape=self._environment.observation_space.shape,
            action_dim=self._environment.action_space.high.size,
            max_action=self._environment.action_space.high,
            lr_actor=generator_params["LearningRateActor", "", 0.001],
            lr_critic=generator_params["LearningRateCritic", "", 0.001],
            actor_units=generator_params["ActorFcLayerParams", "", [
                400, 300]],
            critic_units=generator_params["CriticJointFcLayerParams", "", [
                400, 300]],
            sigma=generator_params["Sigma", "", 0.1],
            tau=generator_params["Tau", "", 0.005],
            n_warmup=generator_params["WarmUp", "", 1000],
            batch_size=generator_params["BatchSize", "", 100],
            gpu=self._params["ML"]["Settings"]["GPUUse", "", 0])
        return policy

    def _get_discriminator(self):
        """Returns instantiated discriminator network -
        parameters from ./examples/example_params/gail_params.json
        """
        local_params = self._params["ML"]["BehaviorGAILAgent"]
        discriminator_params = local_params["Discriminator"]

        irl = GAIL(
            state_shape=self._environment.observation_space.shape,
            action_dim=self._environment.action_space.high.size,
            units=discriminator_params["FcLayerParams", "", [
                400, 300]],
            lr=discriminator_params["LearningRate", "", 0.001],
            enable_sn=local_params["EnableSN", "", False],
            batch_size=discriminator_params["BatchSize", "", 32],
            gpu=self._params["ML"]["Settings"]["GPUUse", "", 0])
        return irl

    def Reset(self):
        """Currently ommited due to missing example in SAC
        """
        pass

    def Act(self, state):
        """Returns action corresponding to state
        """
        return self.generator.get_action(state)

    def Plan(self, observed_world, dt):
        """Currently ommited due to missing example in SAC
        """
        pass

    @property
    def action_space(self):
        """Attribute additionally needed
        """
        return self._environment.action_space
