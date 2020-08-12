


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
)

cc_test(
    name="model_loader_test",
    srcs=["model_loader_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps=[
        "//bark_ml/model_loader:ModelLoader",
        "@libtensorflow-RL-MCTS//libtensorflow:tensorflowlib",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name="observer_test",
    srcs=["observer_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps=[
        "//bark_ml/observers:cpp_observers",
        "@bark_project//bark/world/tests:make_test_world",
        "@gtest//:gtest_main",
    ],
)