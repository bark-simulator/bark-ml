py_library(
    name = "networks",
    srcs = ["__init__.py",
            "gnn_actor_network.py",
            "gnn_actor_distribution_network.py",
            "gnn_value_network.py",
            "gnn_critic_network.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/library_wrappers/lib_tf_agents/networks/gnns:graph_networks"],
    visibility = ["//visibility:public"],
)