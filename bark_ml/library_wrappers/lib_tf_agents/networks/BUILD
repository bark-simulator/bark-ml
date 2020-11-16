py_library(
    name = "networks",
    srcs = ["__init__.py",
            "gnn_actor_network.py",
            "gnn_actor_distribution_network.py",
            "gnn_value_network.py",
            "gnn_critic_network.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = [":graph_networks"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "graph_networks",
    srcs = ["__init__.py",
            "graph_network.py",
            "gnn_gsnt_wrapper.py",
            "gnn_tf2_edge_graph_network.py",
            "gnn_edge_conditioned_wrapper.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = [],
    visibility = ["//visibility:public"],
)