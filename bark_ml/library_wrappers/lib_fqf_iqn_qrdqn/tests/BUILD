py_test(
    name = "save_load_test",
    srcs = ["save_load_test.py"],
    data = [
            "@bark_project//bark:generate_core"
            ],
    imports = ["../external/bark_project/bark/python_wrapper/",
              ],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
            "//bark_ml/commons:py_spaces"
           ],
    visibility = ["//visibility:public"],
)

py_library(
   name = "test_demo_behavior",
   srcs = ["test_demo_behavior.py"]
)

py_test(
    name = "demonstration_collector_test",
    srcs = [
            "demonstration_collector_test.py"
            ],
    data = [
            "@bark_project//bark:generate_core"
            ],
    imports = ["../external/bark_project/bark/python_wrapper/",
              ],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent/demonstrations:demonstrations",
            "//bark_ml/commons:py_spaces",
            ":test_demo_behavior"
           ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_training_benchmark",
    srcs = ["test_training_benchmark.py"],
    data = [
        "//bark_ml:generate_core",
        "@bark_project//bark:generate_core",
    ],
    deps = [
        "@bark_ml_project//bark_ml/environments:blueprints",
        "@bark_ml_project//bark_ml/environments:gym",
        "@bark_ml_project//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
        "@bark_project//bark/runtime/viewer:py_viewer",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer"
    ],
)

py_test(
    name = "test_imitation_agent",
    srcs = ["test_imitation_agent.py"],
    data = [
        "//bark_ml:generate_core",
        "@bark_project//bark:generate_core",
    ],
    deps = [
        "//bark_ml/environments:blueprints",
        "//bark_ml/environments:gym",
        "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
        "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent/demonstrations:demonstrations",
        "@bark_project//bark/runtime/viewer:py_viewer",
        "@benchmark_database//load:benchmark_database",
        "@benchmark_database//serialization:database_serializer"
    ],
)