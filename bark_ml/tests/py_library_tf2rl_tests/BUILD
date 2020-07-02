
py_test(
    name = "load_expert_trajectories_tests",
    srcs = [ "load_expert_trajectories_tests.py"],
    data = [ "//bark_ml/tests/py_library_tf2rl_tests/data:tf2rl_tests_data" ],
    imports = ["../../external/bark_project/bark/python_wrapper/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl:load_expert_trajectories",
            ],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "load_save_utils_tests",
    srcs = ["load_save_utils_tests.py"],
    deps = ["//bark_ml/library_wrappers/lib_tf2rl:load_save_utils"],
    visibility = ["//visibility:public"],
    size = "small"
)


py_test(
    name = "base_tests",
    srcs = ["base_tests.py"],
    imports = ["../../../external/bark_project/bark/python_wrapper/"],

    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl:generate_expert_trajectories",
        ],
    data = [
            "//bark_ml/tests/py_library_tf2rl_tests/data:generate_expert_trajectories_tests_data"
        ],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "simulation_based_tests",
    srcs = ["simulation_based_tests.py"],
    deps = [
            "//bark_ml/tests/py_library_tf2rl_tests:base_tests",
        ],
    visibility = ["//visibility:public"],
    size = "small"
)

# Only run with bazel run //bark_ml/tests/py_library_tf2rl_tests/generate_expert_trajectories_tests:rendered_tests
py_test(
    name = "rendered_tests",
    srcs = ["rendered_tests.py"],
    deps = [
            "//bark_ml/tests/py_library_tf2rl_tests:simulation_based_tests",
        ],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "py_gail_runner_tests",
    srcs = ["py_gail_runner_tests.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so",
            "//bark_ml/tests/py_library_tf2rl_tests/gail_data",
            ],
    imports = ["../../external/bark_project/bark/python_wrapper/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl/runners",
            "@bark_project//bark/runtime:runtime",
            ],
    tags=["no-sandbox"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_gail_agent_tests",
    srcs = ["py_gail_agent_tests.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so",
            "//bark_ml/tests/py_library_tf2rl_tests/gail_data",
            ],
    imports = ["../../external/bark_project/bark/python_wrapper/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl/agents",
            "@bark_project//bark/runtime:runtime",
            ],
    tags=["no-sandbox"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_gail_gym_training_tests",
    srcs = ["py_gail_gym_training_tests.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so",
            "//bark_ml/tests/py_library_tf2rl_tests/gail_data",
            ],
    imports = ["../../external/bark_project/bark/python_wrapper/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl/runners",
            "//bark_ml/library_wrappers/lib_tf2rl/agents",
            "@bark_project//bark/runtime:runtime",
            ],
    tags=["no-sandbox"],
    visibility = ["//visibility:public"],
    size = "small",
    timeout = "moderate"
)

py_test(
    name = "py_gail_bark_training_tests",
    srcs = ["py_gail_bark_training_tests.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so",
            "//bark_ml/tests/py_library_tf2rl_tests/gail_data",
            ],
    imports = ["../../external/bark_project/bark/python_wrapper/"],
    deps = [
            "//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:blueprints",
            "//bark_ml/library_wrappers/lib_tf2rl/runners",
            "//bark_ml/library_wrappers/lib_tf2rl/agents",
            ],
    tags=["no-sandbox"],
    visibility = ["//visibility:public"],
    size = "small",
    timeout = "moderate"
)

py_test(
    name = "py_gail_gym_training_generate_expert_trajectories.py",
    srcs = [ "py_gail_gym_training_generate_expert_trajectories.py" ],
)