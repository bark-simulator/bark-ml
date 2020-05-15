py_library(
    name = "blueprints",
    srcs = glob(["blueprint.py",
                 "blueprints/*/*.py",
                 "blueprints/*/*.xodr"
                 "blueprints/*/*.json"]),
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = [],
    visibility = ["//visibility:public"],
)

py_library(
    name = "rl_runtime",
    srcs = ["rl_runtime.py"],
    imports = ["../external/bark_project/python/"],
    deps = ["//bark_ml/environments:blueprints"],
    visibility = ["//visibility:public"],
)