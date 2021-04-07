py_library(
    name = "runners",
    srcs = ["__init__.py",
            "tfa_runner.py",
            "ppo_runner.py",
            "sac_runner.py"],
    data=['@bark_project//bark:generate_core'],
    deps = ["//bark_ml/commons:tracer",
            "@bark_project//bark/runtime/viewer:buffered_viewer"],
    visibility = ["//visibility:public"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
)