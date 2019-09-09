py_library(
    name = "rl_runtime",
    srcs = ["rl_runtime.py"],
    data = ["@bark_project//python:bark.so",
            ":bark_scenarios"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "//src/wrappers:wrappers",
            "//src/evaluators:evaluators"],
    visibility = ["//visibility:public"],
)