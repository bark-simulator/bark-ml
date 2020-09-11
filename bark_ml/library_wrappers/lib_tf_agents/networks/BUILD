py_library(
    name = "networks",
    srcs = ["__init__.py",
            "gnn_wrapper.py",
            "gnn_actor_network.py",
            "gnn_value_network.py",
            "gnn_critic_network.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["@sgnn//simple_gnn:simple_gnn"],
    visibility = ["//visibility:public"],
)