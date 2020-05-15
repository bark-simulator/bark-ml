py_library(
    name = "blueprints",
    srcs = glob(["blueprints/blueprint.py",
                 "blueprints/__init__.py",
                 "blueprints/*/*.py"]),
    data = glob(["blueprints/*/*.xodr",
                  "blueprints/*/*.json"]),
    imports = ["../external/bark_project/python/"],
    deps = ["//modules/observers:observers",
            "//modules/evaluators:evaluators",
            "//modules/behaviors:behaviors"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "single_agent_runtime",
    srcs = ["single_agent_runtime.py"],
    imports = ["../external/bark_project/python/"],
    data = ["@bark_project//python:bark.so"],
    deps = ["//modules/environments:blueprints"],
    visibility = ["//visibility:public"],
)