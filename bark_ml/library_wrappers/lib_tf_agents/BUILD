py_library(
    name = "tfa_wrapper",
    srcs = [
        "tfa_wrapper.py",
        "gnn_wrapper.py", 
        "gnn_actor_network.py"
    ],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "gnn",
    srcs = [
        "gnn_wrapper.py", 
        "gnn_actor_network.py"
    ],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)