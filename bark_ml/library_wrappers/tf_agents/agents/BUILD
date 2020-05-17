py_library(
    name = "agents",
    srcs = ["sac_agent.py",
            "ppo_agent.py",
            "tfa_agent.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["//bark_ml/library_wrappers/tf_agents:tfa_wrapper"],
    visibility = ["//visibility:public"],
)