py_library(
    name = "agents",
    srcs = ["__init__.py",
            "graph_sac_agent.py",
            "sac_agent.py",
            "ppo_agent.py",
            "tfa_agent.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/library_wrappers/lib_tf_agents:tfa_wrapper",
            "//bark_ml/commons:py_spaces",
            "//bark_ml/library_wrappers/lib_tf_agents/networks"],
    visibility = ["//visibility:public"],
)