from collections import OrderedDict

from torch import nn, sigmoid, cat
from torch.optim import Adam, RMSprop

from .imitation_agent import ImitationAgent


def init_weights(m):
  if type(m) == nn.Linear:
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
                         alpha=0.95,
                         eps=0.00001)


class Carin(nn.Module):
  """
  CARIN = Custom ARchitecture ImitatioN
  """
  def __init__(self, num_channels, num_actions, num_value_functions, params):
    super(Carin, self).__init__()
    self.num_channels = num_channels
    self.num_actions = num_actions
    self.num_value_functions = num_value_functions
    self.shared_layer_dims = params["ML"]["ImitationModel"]["EmbeddingDims",
                                                            "",
                                                            [256, 256, 256]]
    self.head_networks_dims = params["ML"]["CarinModel"][
        "MultitaskLearningEmbeddingDims", "", []]
    self.dropout_p = params["ML"]["ImitationModel"]["DropoutProbability", "",
                                                    0]

    shared_layers, last_layer_dim = self.make_shared_layers()
    head_layers = self.make_head_layers(last_layer_dim)

    self.shared_network = nn.Sequential(shared_layers)
    self.shared_network.apply(init_weights)

    self.head_networks = nn.ModuleList()
    for hl in head_layers:
      net = nn.Sequential(hl)
      net.apply(init_weights)
      self.head_networks.append(net)

  def forward(self, states):
    shared_features = self.shared_network(states)
    outputs = [head(shared_features) for head in self.head_networks]
    action_values = cat(outputs, 1)

    if not self.training:
      # Evaluation phase, output values between 0 and 1
      action_values = sigmoid(action_values)
    return action_values

  def make_shared_layers(self):
    """
    Creates layers that are shared among all predictions
    """
    tuple_list = []
    last_dim = self.num_channels
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
