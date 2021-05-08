from torch import nn, sigmoid
from collections import OrderedDict
from bark_ml.core.value_converters import *

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class Imitation(nn.Module):
  def __init__(self,
               num_channels,
               num_actions,
               num_value_functions,
               params):
    super(Imitation, self).__init__()
    self.num_channels = num_channels
    self.num_actions = num_actions
    self.num_value_functions = num_value_functions
    self.value_converter = NNToValueConverterSequential(self.num_actions)
    self.layer_dims = params["ML"]["ImitationModel"]["EmbeddingDims", "", [256, 256, 256]]
    self.droput_p = params["ML"]["ImitationModel"]["DropoutProbability", "", 0]

    self.net = nn.Sequential(self.make_ordered_layer_dict(self.layer_dims))
    self.net.apply(init_weights)

  def make_ordered_layer_dict(self, layer_dims):
      tuple_list = []
      last_dim = self.num_channels
      for idx, layer in enumerate(layer_dims):
          current_dim = layer
          tuple_list.append((f"layer{idx}", nn.Linear(last_dim, current_dim)))
          tuple_list.append((f"relu{idx}", nn.ReLU()))
          if self.droput_p != 0:
            tuple_list.append((f"relu{idx}", nn.Dropout(p=self.droput_p)))
          last_dim = current_dim
      tuple_list.append(("output", nn.Linear(last_dim, self.num_actions*self.num_value_functions)))
      return OrderedDict(tuple_list)

  @property
  def nn_to_value_converter(self):
    return self.value_converter

  def forward(self, states):
    action_values = self.net(states)
    if not self.training:
      # Evaluation phase, output values between 0 and 1
      action_values = sigmoid(action_values)
    return action_values
