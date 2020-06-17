py_library(
    name = "lib_tf2_gnn",
    srcs = [
        "__init__.py",
        "gnn_wrapper.py", 
        "gnn_actor_network.py",
        "gnn_critic_network.py"
    ],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)