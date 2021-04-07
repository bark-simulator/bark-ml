py_library(
    name = "py_bark_environment",
    srcs = ["py_bark_environment.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["@bark_project//bark/runtime:runtime"],
    visibility = ["//visibility:public"],
)