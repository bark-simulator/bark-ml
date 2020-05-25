py_library(
    name = "blueprints",
    srcs = glob(["blueprints/blueprint.py",
                 "blueprints/__init__.py",
                 "blueprints/*/*.py"]),
    data = glob(["blueprints/*/*.xodr",
                 "blueprints/*/*.json"]) +
                ["@bark_project//python:bark.so",
                 "//python:bark_ml_library.so"],
    imports = ["../external/bark_project/python/",
               "../../python/"],
    deps = ["//bark_ml/observers:observers",
            "//bark_ml/evaluators:evaluators",
            "//bark_ml/behaviors:behaviors",
            "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "single_agent_runtime",
    srcs = ["single_agent_runtime.py"],
    imports = ["../external/bark_project/python/"],
    data = ["@bark_project//python:bark.so"],
    deps = ["//bark_ml/environments:blueprints",
            "@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "gym",
    srcs = ["gym.py"],
    imports = ["../external/bark_project/python/"],
    data = ["@bark_project//python:bark.so"],
    deps = ["//bark_ml/environments:blueprints",
            "//bark_ml/environments:single_agent_runtime"],
    visibility = ["//visibility:public"],
)