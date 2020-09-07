py_library(
    name = "agents",
    srcs = ["__init__.py"],
    deps = [
        ":base_agent",
        ":fqf_agent",
        ":iqn_agent",
        ":qrdqn_agent",
    ],
    visibility = ["//visibility:public"],
)


py_library(
    name = "base_agent",
    srcs = ["base_agent.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    deps = [
        "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/memory:init",
    ],
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

