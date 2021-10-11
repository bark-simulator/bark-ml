from collections import OrderedDict
import torch
from torch import nn, sigmoid, cat, unsqueeze, flatten
from torch.optim import Adam, RMSprop

from .imitation_agent import ImitationAgent
from bark_ml.core.value_converters import *

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)
  elif isinstance(m, nn.Conv1d):
    nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)


class CarinAgent(ImitationAgent):
  def __init__(self, *args, **kwargs):
    super(CarinAgent, self).__init__(*args, **kwargs)

  def init_network(self):
    self.online_net = Carin(
        num_channels=self.observer.observation_space.shape[0],
        num_actions=self.num_actions,
        num_value_functions=self.num_value_functions,
        params=self._params).to(self.device)

    self.optim = RMSprop(self.online_net.parameters(),
                         lr=self.learning_rate,
                         weight_decay=self.weight_decay,
                         alpha=0.95,
                         eps=0.00001)


class Carin(nn.Module):
  """
  CARIN = Custom ARchitecture ImitatioN

  Usage of CNN layers inspired by:
    'Combining Planning and Deep Reinforcement Learning in Tactical
    Decision Making for Autonomous Driving' (Hoel et al., 2019)
  """
  def __init__(self, num_channels, num_actions, num_value_functions, params):
    super(Carin, self).__init__()
    self.num_channels = num_channels
    self.num_actions = num_actions
    self.num_value_functions = num_value_functions

    self.num_features_ego = params["ML"]["CarinModel"]["NumFeaturesEgo", "", 6]
    self.num_features_other_agent = params["ML"]["CarinModel"][
        "NumFeaturesOtherAgent", "", 4]
    self.num_other_agents = \
      (self.num_channels - self.num_features_ego) // self.num_features_other_agent

    self.dropout_p = params["ML"]["ImitationModel"]["DropoutProbability", "",
                                                    0]

    self.input_conv_dims = params["ML"]["CarinModel"]["InputConvDims", "", []]
    self.shared_layer_dims = params["ML"]["ImitationModel"]["EmbeddingDims",
                                                            "",
                                                            [256, 256, 256]]
    self.head_networks_dims = params["ML"]["CarinModel"][
        "MultitaskLearningEmbeddingDims", "", []]

    # CNN layers (transform input features of each agent separately).
    input_conv_layers, last_input_layer_dim = self.make_input_conv_layers()
    self.input_conv_net = nn.Sequential(input_conv_layers)
    self.input_conv_net.apply(init_weights)

    # Shared FC layers
    shared_layers, last_shared_layer_dim = self.make_shared_layers(
        last_input_layer_dim)
    self.shared_network = nn.Sequential(shared_layers)
    self.shared_network.apply(init_weights)

    # Multitask learning (separate FC networks for each value function)
    head_layers = self.make_head_layers(last_shared_layer_dim)
    self.head_networks = nn.ModuleList()
    for hl in head_layers:
      net = nn.Sequential(hl)
      net.apply(init_weights)
      self.head_networks.append(net)
    
    self.value_converter = NNToValueConverterSequential(self.num_actions)

  @torch.jit.unused
  @property
  def nn_to_value_converter(self):
    return self.value_converter

  def forward(self, states):
    # Treat ego features and features of other agents separately
    ego_features = states[:, :self.num_features_ego]
    other_agent_features = states[:, self.num_features_ego:]

    # Convert input tensor from 2D to 3D (needed for convolution afterwards)
    other_agent_features = unsqueeze(other_agent_features, 1)

    # Use CNN to transform features of each agent separately
    other_agent_features = self.input_conv_net(other_agent_features)

    # Transform back to 2D
    other_agent_features = flatten(other_agent_features,
                                   start_dim=1,
                                   end_dim=2)

    # Concatenate ego features with the features of other agents
    transformed_input = cat([ego_features, other_agent_features], 1)

    # Fully connected layers
    shared_features = self.shared_network(transformed_input)

    # Compute output for each value function, then concat them
    outputs = [head(shared_features) for head in self.head_networks]
    action_values = cat(outputs, 1)

    if not self.training:
      # Evaluation phase, output values between 0 and 1
      action_values = sigmoid(action_values)
    return action_values

  def make_input_conv_layers(self):
    tuple_list = []
    in_channels = 1
    kernel_size = self.num_features_other_agent
    stride = kernel_size
    for idx, layer in enumerate(self.input_conv_dims):
      out_channels = layer
      tuple_list.append((f"conv_layer{idx}",
                         nn.Conv1d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride)))
      tuple_list.append((f"relu{idx}", nn.ReLU()))
      in_channels = out_channels
      kernel_size = 1
      stride = 1

    if len(self.input_conv_dims) > 0:
      tuple_list.append(
          ("maxpool", nn.MaxPool2d(kernel_size=(1, self.num_other_agents))))
      # Ego features are appended to the end of the output of CNN
      last_dimension = in_channels + self.num_features_ego
    else:
      last_dimension = self.num_channels

    return OrderedDict(tuple_list), last_dimension

  def make_shared_layers(self, start_layer_dim):
    """
    Creates layers that are shared among all predictions
    """
    tuple_list = []
    last_dim = start_layer_dim
    for idx, layer in enumerate(self.shared_layer_dims):
      current_dim = layer
      tuple_list.append((f"layer{idx}", nn.Linear(last_dim, current_dim)))
      tuple_list.append((f"relu{idx}", nn.ReLU()))
      if self.dropout_p != 0:
        tuple_list.append((f"dropout{idx}", nn.Dropout(p=self.dropout_p)))
      last_dim = current_dim
    return OrderedDict(tuple_list), last_dim

  def make_head_layers(self, start_layer_dim):
    """
    Creates separate layers for each value function output (multitask learning)
    """
    head_layers = []
    for i in range(1, self.num_value_functions + 1):
      tuple_list = []
      last_dim = start_layer_dim
      for idx, layer in enumerate(self.head_networks_dims):
        current_dim = layer
        tuple_list.append(
            (f"head{i}_layer{idx}", nn.Linear(last_dim, current_dim)))
        tuple_list.append((f"head{i}_relu{idx}", nn.ReLU()))
        if self.dropout_p != 0:
          tuple_list.append(
              (f"head{i}_dropout{idx}", nn.Dropout(p=self.dropout_p)))
        last_dim = current_dim
      tuple_list.append((f"output{i}", nn.Linear(last_dim, self.num_actions)))
      head_layers.append(OrderedDict(tuple_list))
    return head_layers
