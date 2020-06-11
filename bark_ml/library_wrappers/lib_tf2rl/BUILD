py_library(
    name = "tf2rl_wrapper",
    srcs = ["tf2rl_wrapper.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
)