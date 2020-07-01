


py_test(
    name = "py_environment_tests",
    srcs = ["py_environment_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/",
               "../python_wrapper/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_behavior_tests",
    srcs = ["py_behavior_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/behaviors:behaviors",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_observer_tests",
    srcs = ["py_observer_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so",
            "//bark_ml:generate_core"],
    imports = ["../external/bark_project/bark/python_wrapper/",
               "../python_wrapper/"],
    deps = ["@bark_project//bark/runtime:runtime",
            "//bark_ml/environments:single_agent_runtime",
            "//bark_ml/behaviors:behaviors",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_evaluator_tests",
    srcs = ["py_evaluator_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so",
            "//bark_ml:generate_core"],
    imports = ["../external/bark_project/bark/python_wrapper/",
               "../python_wrapper/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/behaviors:behaviors",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_library_tfa_tests",
    srcs = ["py_library_tfa_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
            "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)
