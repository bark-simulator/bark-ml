from torch import nn
from collections import OrderedDict

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
    self.layer_dims = params["ML"]["FQFModel"]["EmbeddingDims", "", [256, 256, 256]]

    self.net = nn.Sequential(self.make_ordered_layer_dict(self.layer_dims))
    self.net.apply(init_weights)

  def make_ordered_layer_dict(self, layer_dims):
      tuple_list = []
      last_dim = self.num_channels
      for idx, layer in enumerate(layer_dims):
          current_dim = layer
          tuple_list.append((f"layer{idx}", nn.Linear(last_dim, current_dim)))
          tuple_list.append((f"relu{idx}", nn.ReLU()))
          last_dim = current_dim
      tuple_list.append(("output", nn.Linear(last_dim, self.num_actions*self.num_value_functions)))
      return OrderedDict(tuple_list)

  def forward(self, states):
    batch_size = states.shape[0]

    action_values = self.net(states)
    return action_values

