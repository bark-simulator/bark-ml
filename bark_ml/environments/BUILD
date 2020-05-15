py_library(
    name = "blueprints",
    srcs = glob(["blueprints/blueprint.py",
                 "blueprints/*/*.py"]),
    data = glob(["blueprints/*/*.xodr",
                  "blueprints/*/*.json"]),
    imports = ["../external/bark_project/python/"],
    deps = [],
    visibility = ["//visibility:public"],
)

py_library(
    name = "single_agent_runtime",
    srcs = ["single_agent_runtime.py"],
    imports = ["../external/bark_project/python/"],
    deps = ["//bark_ml/environments:blueprints"],
    visibility = ["//visibility:public"],
)