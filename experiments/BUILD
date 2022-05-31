filegroup(
    name = "data",
    srcs = glob(["*.json", "**/*.yaml", "**/*.json", "**/*.xodr", "**/*.py"]),
    visibility = ["//visibility:public"],
)

py_binary(
    name = "run_experiment",
    srcs = ["run_experiment.py"],
    data = ["@bark_project//bark/python_wrapper:core.so",
            "//bark_ml:generate_core",
            ":data"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/experiment:experiment_runner"],
    visibility = ["//visibility:public"],
)