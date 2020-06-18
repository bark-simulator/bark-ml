py_library(
    name = "agents",
    srcs = ["__init__.py",
            "sac_agent.py",
            "ppo_agent.py",
            "tfa_agent.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/library_wrappers/lib_tf_agents:tfa_wrapper",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
)