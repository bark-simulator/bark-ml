filegroup(
    name = "data",
    srcs = glob(["**/*.yaml", "**/*.json", "**/*.xodr", "**/*.py"]),
    visibility = ["//visibility:public"],
)


py_library(
    name = "experiment_template",
    srcs = ["experiment_template.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_experiment_tests",
    srcs = ["py_experiment_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so",
            ":data"],
    imports = ["../external/bark_project/bark/python_wrapper/",
               "../python_wrapper/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:counterfactual_runtime",
            "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
            "//bark_ml/environments:gym",
            "//bark_ml/commons:py_spaces",
            ":experiment_template"],
    visibility = ["//visibility:public"],
)