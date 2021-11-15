py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/environments:environments",
            "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
            "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
            "//bark_ml/evaluators:evaluators",
            "//bark_ml/observers:observers"],
    visibility = ["//visibility:public"],
)

py_binary(
    name = "experiment_runner",
    srcs = ["experiment_runner.py"],
    data = ["@bark_project//bark/python_wrapper:core.so",
            "//bark_ml:generate_core"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = [":experiment"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_experiment_tests",
    srcs = ["py_experiment_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/",
               "../python_wrapper/"],
    deps = [":experiment"],
    visibility = ["//visibility:public"]
)

