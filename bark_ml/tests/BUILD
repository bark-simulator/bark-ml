filegroup(
    name = "data",
    srcs = glob(["data/*.yaml", "data/*.json", "data/*.xodr"]),
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_environment_tests",
    srcs = ["py_environment_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so",
            ":data"],
    imports = ["../external/bark_project/bark/python_wrapper/",
               "../python_wrapper/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:counterfactual_runtime",
            "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
            "//bark_ml/environments:gym",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_apollo_tests",
    srcs = ["py_apollo_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so",
            ":data"],
    imports = ["../external/bark_project/bark/python_wrapper/",
               "../python_wrapper/"],
    deps = ["//bark_ml/environments:external_runtime",
            "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
            "//bark_ml/environments:single_agent_runtime",
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

py_test(
    name = "py_bark_behavior_model_tests",
    srcs = ["py_bark_behavior_model_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
            "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_graph_observer_tests",
    srcs = ["py_graph_observer_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/behaviors:behaviors",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
)


cc_test(
    name = "observer_nearest_test",
    srcs = [
        "observer_nearest_test.cc",
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "@bark_project//bark/geometry",
        "@bark_project//bark/world:world",
        "@bark_project//bark/world/map:map_interface",
        "@bark_project//bark/world/map:roadgraph",
        "@bark_project//bark/world/opendrive:opendrive",
        "@bark_project//bark/models/behavior/constant_acceleration:constant_acceleration",
        "@bark_project//bark/models/execution/interpolation:interpolation",
        "@bark_project//bark/world/evaluation:evaluation",
        "@bark_project//bark/models/behavior/motion_primitives:motion_primitives",
        "@bark_project//bark/world/tests:make_test_world",
        "//bark_ml/observers:cpp_observers",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "observer_static_test",
    srcs = [
        "observer_static_test.cc",
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "@bark_project//bark/geometry",
        "@bark_project//bark/world:world",
        "@bark_project//bark/world/map:map_interface",
        "@bark_project//bark/world/map:roadgraph",
        "@bark_project//bark/world/opendrive:opendrive",
        "@bark_project//bark/models/behavior/constant_acceleration:constant_acceleration",
        "@bark_project//bark/models/execution/interpolation:interpolation",
        "@bark_project//bark/world/evaluation:evaluation",
        "@bark_project//bark/models/behavior/motion_primitives:motion_primitives",
        "@bark_project//bark/world/tests:make_test_world",
        "//bark_ml/observers:cpp_observers",
        "@gtest//:gtest_main",
    ],
)
