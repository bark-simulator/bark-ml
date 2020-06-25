
py_test(
    name = "load_expert_trajectories_tests",
    srcs = [ "load_expert_trajectories_tests.py"],
    data = [ "//bark_ml/tests/py_library_tf2rl_tests/data:tf2rl_tests_data" ],
    imports = [
            "../../external/bark_project/python/"
            ],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl:load_expert_trajectories",
            ],
    visibility = ["//visibility:public"],
    size = "small"
)

py_test(
    name = "generate_expert_trajectories_tests",
    srcs = ["generate_expert_trajectories_tests.py"],
    imports = [
            "../../external/bark_project/python/"
        ],
    deps = [
            "//bark_ml/library_wrappers/lib_tf2rl:generate_expert_trajectories",
        ],
    data = [
            "//bark_ml/tests/py_library_tf2rl_tests/data:tf2rl_tests_data"
        ],
    visibility = ["//visibility:public"],
    size = "small"
)