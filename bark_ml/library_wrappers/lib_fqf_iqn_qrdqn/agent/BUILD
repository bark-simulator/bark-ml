py_library(
    name = "agents",
    srcs = ["__init__.py"],
    deps = [
        ":base_agent",
        ":fqf_agent",
        ":iqn_agent",
        ":qrdqn_agent",
        ":imitation_agent",
        ":carin_agent",
        "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:util",
        "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent/demonstrations:demonstrations",
    ],
    visibility = ["//visibility:public"],
)


py_library(
    name = "util",
    srcs = [
        "util.py"
    ],
    visibility = ["//visibility:public"],
)


py_library(
    name = "base_agent",
    srcs = ["base_agent.py"
    ],
    data = [
        "@bark_project//bark:generate_core"
        ],
    deps = [
        "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/memory:init",
        "//bark_ml/behaviors:behaviors",
    ],
)

py_library(
    name = "imitation_agent",
    srcs = ["imitation_agent.py"],
    deps = [":base_agent",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent/loss:loss_function",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model:model",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn:utils",
    ],
    visibility = ["//visibility:public"],
)


py_library(
    name = "fqf_agent",
    srcs = ["fqf_agent.py"],
    deps = [":base_agent",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model:model",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn:utils",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "iqn_agent",
    srcs = ["iqn_agent.py"],
    deps = [":base_agent",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model:model",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn:utils",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "qrdqn_agent",
    srcs = ["qrdqn_agent.py"],
    deps = [":base_agent",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model:model",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn:utils",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "carin_agent",
    srcs = ["carin_agent.py"],
    deps = [":imitation_agent"],
    visibility = ["//visibility:public"],
)
