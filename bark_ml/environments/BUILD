py_library(
    name = "blueprints",
    srcs = glob(["blueprints/blueprint.py",
                 "blueprints/__init__.py",
                 "blueprints/*/*.py"]),
    data = glob(["blueprints/*/*.xodr",
                 "blueprints/*/*.json",
                 "blueprints/*.json"]) +
                ["@bark_project//bark/python_wrapper:core.so",
                 "//bark_ml:generate_core"],
    imports = ["../external/bark_project/bark/python_wrapper/",
               "../python_wrapper/"],
    deps = ["//bark_ml/observers:observers",
            "//bark_ml/evaluators:evaluators",
            "//bark_ml/behaviors:behaviors",
            "@bark_project//bark/runtime/scenario/scenario_generation:scenario_generation"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "single_agent_runtime",
    srcs = ["single_agent_runtime.py"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    deps = ["//bark_ml/environments:blueprints",
            "@bark_project//bark/runtime:runtime"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "counterfactual_runtime",
    srcs = ["counterfactual_runtime.py"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    deps = ["//bark_ml/environments:blueprints",
            "@bark_project//bark/runtime:runtime",
            ":single_agent_runtime",
            "//bark_ml/commons:tracer"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "gym",
    srcs = ["gym.py"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    deps = ["//bark_ml/environments:blueprints",
            "//bark_ml/environments:single_agent_runtime"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "environments",
    srcs = ["__init__.py"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    deps = ["//bark_ml/environments:blueprints",
            "//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:counterfactual_runtime"],
    visibility = ["//visibility:public"],
)