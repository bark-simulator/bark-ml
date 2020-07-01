
py_test(
    name = "base_tests",
    srcs = ["base_tests.py"],
    imports = ["../../../external/bark_project/bark/python_wrapper/"],

    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl:generate_expert_trajectories",
        ],
    data = [
            "//bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests/data:generate_expert_trajectories_tests_data"
        ],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "simulation_based_tests",
    srcs = ["simulation_based_tests.py"],
    deps = [
            "//bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests:base_tests",
        ],
    visibility = ["//visibility:public"],
    size = "small"
)

# Only run with bazel run //bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests:rendered_tests
py_test(
    name = "rendered_tests",
    srcs = ["rendered_tests.py"],
    deps = [
            "//bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests:simulation_based_tests",
        ],
    visibility = ["//visibility:public"],
    size = "small"
)