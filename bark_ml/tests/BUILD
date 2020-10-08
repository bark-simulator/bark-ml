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

py_test(
    name = "py_library_torch_agents_tests",
    srcs = ["py_library_torch_agents_tests.py"],
    data = ["@torchcpp//:lib",
            "@bark_project//bark/python_wrapper:core.so",
            "//bark_ml/tests/lib_fqf_iqn_qrdqn_test_data:weights",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model_wrapper:pytorch_script_wrapper.so"],
    imports = ["../external/bark_project/bark/python_wrapper/",
               "../external/torchcpp/torch/lib" # this path does not exist even after adding data dependency. Keeping it for further research. 
                                                # and ../ should be ../../ for bark as well as pytorch as there is no external 
                                                # in runfiles/bark_ml/bark_ml/external (not exists but the import adds it!)
              ],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
            "//bark_ml/commons:py_spaces"
           ],
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

py_test(
    name = "py_gnn_wrapper_tests",
    srcs = ["py_gnn_wrapper_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so",
            "//bark_ml:generate_core"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = [
        "//bark_ml/environments:single_agent_runtime",
        "//bark_ml/environments:blueprints",
        "//bark_ml/library_wrappers/lib_tf_agents/networks:networks",
        "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
        "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_tracer_tests",
    srcs = ["py_tracer_tests.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
            "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
            "//bark_ml/commons:py_spaces",
            "//bark_ml/commons:tracer"],
    visibility = ["//visibility:public"],
)
