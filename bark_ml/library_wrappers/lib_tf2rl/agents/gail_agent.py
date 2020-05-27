import sys
import logging
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

# tf2rl imports
import tf2rl

# BARK imports
from bark_ml.library_wrappers.lib_tf2rl.agents.tf2rl_agent import BehaviorTF2RLAgent
from bark_ml.behaviors.cont_behavior import BehaviorContinuousML

class BehaviorGAILAgent(BehaviorTF2RLAgent, BehaviorContinuousML):
    """GAIL agent based on the tf2rl library."""

    def __init__(self,
                 environment=None,
                 params=None):
        
        BehaviorTF2RLAgent.__init__(environment=environment,
                                    params=params)
        BehaviorContinuousML.__init__(self, params)

        self._generator = self._get_generator()
        self._discriminator = self._get_discriminator()


    def _get_generator(self):
        """Instantiate DDPG generator here."""
        pass

    
    def _get_discriminator(self):
        """Instantiate discriminator network here."""


    def Reset(self):
        """Has to be implemented here"""
        pass


    def Act(self, state):
        """Has to be implemented here"""
        pass


    def Plan(self, observed_world, dt):
        """Has to be implemented here"""
        pass


