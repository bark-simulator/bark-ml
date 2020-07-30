py_library(
    name = "runners",
    srcs = ["__init__.py",
            "gail_runner.py",
            "tf2rl_runner.py"],
    deps = ["@com_github_keiohta_tf2rl",
    "//bark_ml/library_wrappers/lib_tf2rl:compare_trajectories"],
    visibility = ["//visibility:public"],
)