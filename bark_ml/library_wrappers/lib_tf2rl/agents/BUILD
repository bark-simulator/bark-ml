py_library(
    name = "agents",
    srcs = ["__init__.py",
            "gail_agent.py",
            "tf2rl_agent.py",],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["//bark_ml/library_wrappers/lib_tf2rl:tf2rl_wrapper",
            "//bark_ml/commons:py_spaces",
            "//bark_ml/behaviors"],
    visibility = ["//visibility:public"],
)