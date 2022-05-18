from bark_ml.library_wrappers.lib_tf_agents.networks.gnns.interaction_wrapper import InteractionWrapper

def init_interaction_network(name, params):
  return InteractionWrapper(
    params=params,
    name=name + "_InteractionNetwork")
