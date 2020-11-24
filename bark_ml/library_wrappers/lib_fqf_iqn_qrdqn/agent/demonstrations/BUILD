py_library(
    name = "demonstrations",
    srcs = ["demonstration_collector.py",
            "__init__.py"
            ],
    deps = [
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:training_benchmark_database",
            "@bark_project//bark/benchmark:benchmark_runner",
            "@bark_project//bark/benchmark:benchmark_runner_mp",
            "@bark_project//bark/benchmark:benchmark_result",
    ],
    visibility = ["//visibility:public"],
)

