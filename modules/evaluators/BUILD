py_library(
    name = "evaluators",
    srcs = ["goal_reached.py",
            "evaluator.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)