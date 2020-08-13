
py_test(
    name = "load_expert_trajectories_tests",
    srcs = [ "load_expert_trajectories_tests.py"],
    data = [ 
        "//bark_ml/tests/py_library_tf2rl_tests/data:generate_expert_trajectories_tests_data",
        "@bark_project//bark/python_wrapper:core.so",
        "//bark_ml/tests/py_library_tf2rl_tests/gail_data",
        "//bark_ml/observers",
        ],
    imports = ["../../bark_ml/python_wrapper/"],
    deps = [
            "@com_github_keiohta_tf2rl",
            "//bark_ml/library_wrappers/lib_tf2rl:load_expert_trajectories",
            "@bark_project//bark/runtime:runtime",
            ],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "normalize_on_merging_blueprint_tests",
    srcs = [ "normalize_on_merging_blueprint_tests.py"],
    data = [ 
        "//bark_ml/tests/py_library_tf2rl_tests/gail_data",
        ],
    imports = ["../../bark_ml/python_wrapper/"],
    deps = [
            "generate_sac_trajectory_tests_base",
            "@com_github_keiohta_tf2rl",
            "//bark_ml/library_wrappers/lib_tf2rl",
            ],
    visibility = ["//visibility:public"],
    size = "small"
)

py_library(
    name = "generate_sac_trajectory_tests_base",
    srcs = [ "generate_sac_trajectory_tests_base.py"],
    data = [ 
        "//bark_ml/tests/py_library_tf2rl_tests/data:generate_expert_trajectories_tests_data",
        "@bark_project//bark/python_wrapper:core.so",
        "@com_github_gail_4_bark_large_data_store",
        ],
    imports = ["../../bark_ml/python_wrapper/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl:load_expert_trajectories",
            "@bark_project//bark/runtime:runtime",
            "//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:blueprints",
            "//bark_ml/library_wrappers/lib_tf_agents/runners",
            "//bark_ml/library_wrappers/lib_tf_agents/agents",
            "//bark_ml/library_wrappers/lib_tf_agents:tfa_wrapper",
            ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "load_save_utils_tests",
    srcs = ["load_save_utils_tests.py"],
    deps = ["//bark_ml/library_wrappers/lib_tf2rl:load_save_utils"],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "normalization_utils_tests",
    srcs = ["normalization_utils_tests.py"],
    deps = ["//bark_ml/library_wrappers/lib_tf2rl:normalization_utils"],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "compare_trajectories_tests",
    srcs = ["compare_trajectories_tests.py"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl:compare_trajectories"],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "base_tests",
    srcs = ["base_tests.py"],
    imports = ["../../../bark_ml/python_wrapper/"],

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
    name = "calculate_action_tests",
    srcs = ["calculate_action_tests.py"],
    imports = ["../../../bark_ml/python_wrapper/"],
    deps = [
            "generate_sac_trajectory_tests_base",
            "//bark_ml/library_wrappers/lib_tf2rl:lib_tf2rl",
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
    size = "small",
)

# Only run with 
py_binary(
    name = "rendered_tests",
    srcs = ["rendered_tests.py"],
    deps = [
            "//bark_ml/tests/py_library_tf2rl_tests:simulation_based_tests",
        ],
    visibility = ["//visibility:public"],
    testonly = True
)

py_test(
    name = "py_gail_runner_tests",
    srcs = ["py_gail_runner_tests.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so",
            "//bark_ml/tests/py_library_tf2rl_tests/gail_data",
            ],
    imports = ["../../bark_ml/python_wrapper/"],
    deps = [
            "//bark_ml/environments:blueprints",
            "//bark_ml/environments:single_agent_runtime",
            "//bark_ml/library_wrappers/lib_tf2rl/runners",
            "//bark_ml/library_wrappers/lib_tf2rl/agents",
            "//bark_ml/library_wrappers/lib_tf2rl:tf2rl_wrapper",
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
    imports = ["../../bark_ml/python_wrapper/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl/agents",
            "@bark_project//bark/runtime:runtime",
            ],
    tags=["no-sandbox"],
    visibility = ["//visibility:public"],
    size = "small",
)

py_test(
    name = "py_gail_training_tests",
    srcs = ["py_gail_training_tests.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so",
            "//bark_ml/tests/py_library_tf2rl_tests/gail_data",
            ],
    imports = ["../../bark_ml/python_wrapper/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl/runners",
            "//bark_ml/library_wrappers/lib_tf2rl/agents",
            ],
    tags=["no-sandbox"],
    visibility = ["//visibility:public"],
    size = "small",
    timeout = "moderate"
)

py_test(
    name = "py_tf2rl_wrapper_tests",
    srcs = ["py_tf2rl_wrapper_tests.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so",
            ],
    imports = ["../../bark_ml/python_wrapper/"],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl:tf2rl_wrapper",
            ],
    visibility = ["//visibility:public"],
    size = "small",
)