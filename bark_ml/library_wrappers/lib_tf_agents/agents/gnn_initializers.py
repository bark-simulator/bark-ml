from bark_ml.library_wrappers.lib_tf_agents.networks.graph_network import GraphNetwork
from bark_ml.library_wrappers.lib_tf_agents.networks.gnn_gsnt_wrapper import GSNTWrapper


def init_gsnt(name, params):
  return GSNTWrapper(
    params=params, 
    name=name + "_GSNT")
  
def init_gnn_edge_cond(name, params):
  return GEdgeCondWrapper(
    params=params, 
    name=name + "_GEdgeCond")