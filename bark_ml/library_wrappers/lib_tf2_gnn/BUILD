py_library(
    name = "lib_tf2_gnn",
    srcs = [
        "__init__.py",
        "gnn_wrapper.py", 
        "gnn_actor_network.py",
        "gnn_critic_network.py"
    ],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["@bark_project//bark/runtime:runtime"],
    visibility = ["//visibility:public"],
)