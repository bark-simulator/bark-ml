py_test(
    name = "py_environment_tests",
    srcs = ["py_environment_tests.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/",
               "../../python/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_behavior_tests",
    srcs = ["py_behavior_tests.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/behaviors:behaviors",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_observer_tests",
    srcs = ["py_observer_tests.py"],
    data = ["@bark_project//python:bark.so",
            "//python:bark_ml_library.so"],
    imports = ["../external/bark_project/python/",
               "../../python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "//bark_ml/environments:single_agent_runtime",
            "//bark_ml/behaviors:behaviors",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_evaluator_tests",
    srcs = ["py_evaluator_tests.py"],
    data = ["@bark_project//python:bark.so",
            "//python:bark_ml_library.so"],
    imports = ["../external/bark_project/python/",
               "../../python/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/behaviors:behaviors",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_library_tfa_tests",
    srcs = ["py_library_tfa_tests.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
            "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_library_tf2rl_tests",
    srcs = ["py_library_tf2rl_tests.py"],
    data = ["data/mock_expert_trajectories.pkl"],
    imports = ["../external/bark_project/python/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl:utils",
            ],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "py_gail_runner_tests",
    srcs = ["py_gail_runner_tests.py"],
    data = [
            "@bark_project//python:bark.so",
            "gail_testfiles",
            "gail_testdirs",
            ],
    imports = ["../external/bark_project/python/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl/runners",
            "@bark_project//modules/runtime:runtime",
            ],
    visibility = ["//visibility:public"],
    size = "small",
)

filegroup(
    name = "gail_testfiles",
    srcs = glob([
                "gail_data/*",
                ]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "gail_testdirs",
    srcs = ["gail_data"],
    visibility = ["//visibility:public"],
)