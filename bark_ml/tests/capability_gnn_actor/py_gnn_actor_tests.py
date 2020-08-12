# Copyright (c) 2020 fortiss GmbH
#
# Authors: Silvan Wimmer
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import unittest
import numpy as np

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import ContinuousHighwayBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
from bark_ml.library_wrappers.lib_tf_agents.agents import BehaviorGraphSACAgent
from bark_ml.observers.graph_observer import GraphObserver

# Supervised specific imports
from bark_ml.tests.capability_gnn_actor.actor_nets import ConstantActorNet,\
  RandomActorNet
from bark_ml.tests.capability_gnn_actor.data_handler import SupervisedData
from bark_ml.tests.capability_gnn_actor.learner import Learner

class PyGNNActorTests(unittest.TestCase):

  def setUp(self):
    """Setting up the test cases"""
    # Parameter
    self._log_dir = None 
    self._epochs = 300
    self._batch_size = 32
    self._train_split = 0.8
    self._data_path = "bark_ml/tests/capability_gnn_actor/data"
    self._num_scenarios = 2

    # Filenames for parameter files
    filename_tf2_gnn = "examples/example_params/tfa_sac_gnn_spektral_default.json"
    filename_spektral = "examples/example_params/tfa_sac_gnn_tf2_gnn_default.json"

    # Setting up tf2_gnn actor net
    params, observer, actor = self._configurable_setup(filename_tf2_gnn)
    self._params_tf2_gnn = params
    self._observer_tf2_gnn = observer
    self._actor_tf2_gnn = actor

    # Setting up spektral actor net
    params, observer, actor = self._configurable_setup(filename_spektral)
    self._params_spektral = params
    self._observer_spektral = observer
    self._actor_spektral = actor
  
  def test_existence_actor_networks(self):
    """Evaluates existence of tf2_gnn actor and spektral actor"""
    self.assertIsNotNone(self._actor_tf2_gnn)
    self.assertIsNotNone(self._actor_spektral)
  
  def test_spektral_actor_overfitting(self):
    """Checks if spektral actor is capable of overfitting anything by comparing
    it on a very small dataset with a random actor and an actor outputting
    the mean of the dataset.
    
    This test trains an GNN SAC agent and considers the average
    of train losses per epoch for the last 20 epochs for comparison. This
    average is then compared with the average train losses of Random and 
    Constant actors.
    """
    losses = self._overfitting_test(self._actor_spektral, self._observer_spektral, self._params_spektral)

    # Compare the losses (means)
    avg_train_loss_constant = np.mean(np.array(losses["constant_actor"]["train"]))
    avg_train_loss_random = np.mean(np.array(losses["random_actor"]["train"]))

    # Consider only last 20 epochs of GNN for comparison
    avg_train_loss_gnn = np.mean(np.array(losses["gnn_actor"]["train"][-20:]))
    print("Spektral actor overfitting test results:")
    print("avg_train_loss_constant:", avg_train_loss_constant)
    print("avg_train_loss_random:", avg_train_loss_random)
    print("avg_train_loss_spektral:", avg_train_loss_gnn, "\n\n")

    self.assertLess(avg_train_loss_gnn, avg_train_loss_random)
    self.assertLess(avg_train_loss_gnn, avg_train_loss_constant)

  def test_tf2_gnn_actor_overfitting(self):
    """Checks if TF2_GNN actor is capable of overfitting anything by comparing
    it on a very small dataset with a random actor and an actor outputting
    the mean of the dataset.
    
    This test trains an GNN SAC agent  and considers the average
    of train losses per epoch for the last 20 epochs for comparison. This
    average is then compared with the average train losses of Random and 
    Constant actors.
    """
    losses = self._overfitting_test(self._actor_tf2_gnn, self._observer_tf2_gnn, self._params_tf2_gnn)

    # Compare the losses (means)
    avg_train_loss_constant = np.mean(np.array(losses["constant_actor"]["train"]))
    avg_train_loss_random = np.mean(np.array(losses["random_actor"]["train"]))

    # Consider only last 20 epochs of GNN for comparison
    avg_train_loss_gnn = np.mean(np.array(losses["gnn_actor"]["train"][-20:]))
    print("tf2_gnn actor overfitting test results:")
    print("avg_train_loss_constant:", avg_train_loss_constant)
    print("avg_train_loss_random:", avg_train_loss_random)
    print("avg_train_loss_tf2_gnn:", avg_train_loss_gnn, "\n\n")

    self.assertLess(avg_train_loss_gnn, avg_train_loss_random)
    self.assertLess(avg_train_loss_gnn, avg_train_loss_constant)
    
  def _configurable_setup(self, params_filename):
    """Configurable GNN setup depending on a given filename

    Args:
      params_filename: str, corresponds to path of params file

    Returns:
      params: ParameterServer instance
      observer: GraphObserver instance
      actor: ActorNetwork of BehaviorGraphSACAgent
    """
    params = ParameterServer(filename=params_filename)
    observer = GraphObserver(params=params)
    bp = ContinuousHighwayBlueprint(params,
                                    number_of_senarios=2,
                                    random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, observer=observer,
                             render=False)
    # Get GNN SAC actor net
    sac_agent = BehaviorGraphSACAgent(environment=env, observer=observer,
                                      params=params)
    actor = sac_agent._agent._actor_network
    return params, observer, actor

  def _overfitting_test(self, actor, observer, params):
    """Tests if actor can be overfit to small dataset.
    
    Generates small dataset. Runs benchmark tests with a constant and a random
    actor. And compares them to the actor which is given as function parameter.
    
    Args:
      actor: Actor of BehaviorGraphSACAgent
      observer: GraphObserver instance
      params: ParameterServer instance

    Returns:
      losses: Dict, containing all losses (of "constant_actor", "random_actor", and "gnn_actor")
    """
     # Build dataset
    dataset = SupervisedData(observer, params,
                             data_path=self._data_path,
                             batch_size=self._batch_size,
                             train_split=self._train_split,
                             num_scenarios=self._num_scenarios)
    train_dataset = dataset._train_dataset
    test_dataset = dataset._test_dataset

    # Get benchmark data (run Constant Actor)
    constant_actor = ConstantActorNet(dataset=train_dataset)
    learner1 = Learner(constant_actor, train_dataset, test_dataset)
    losses_constant = learner1.train(epochs=20, only_test=True,
                                           mode="Number")
    # Get benchmark data (run RandomActor)
    random_actor = RandomActorNet(low=[0, -0.4], high=[0.1, 0.4])
    learner2 = Learner(random_actor, train_dataset, test_dataset)
    losses_random = learner2.train(epochs=20, only_test=True,
                                         mode="Number")
    # Learn agent
    learner = Learner(actor, train_dataset, test_dataset,
                      log_dir=self._log_dir)
    losses_gnn = learner.train(epochs=self._epochs, only_test=False,
                               mode="Distribution")
    losses = dict()
    losses["random_actor"] = losses_random
    losses["constant_actor"] = losses_constant
    losses["gnn_actor"] = losses_gnn
    return losses
    
                                        
if __name__ == '__main__':
    unittest.main()
