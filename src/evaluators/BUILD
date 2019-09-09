py_library(
    name = "evaluators",
    srcs = ["state_evaluator.py"],
    data = ["@bark_project//python:bark.so",
            ":bark_scenarios"],
    imports = ["../external/bark_project/python/"],
		deps = ["@bark_project//modules/runtime:runtime"],
		visibility = ["//visibility:public"],
)