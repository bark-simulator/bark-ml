py_library(
    name = "model",
    srcs = ["__init__.py"],
    deps = [
        ":base_model",
        ":fqf",
        ":iqn",
        ":qrdqn",
    ],
    visibility = ["//visibility:public"],
)


py_library(
    name = "base_model",
    srcs = ["base_model.py"],
    deps = ["//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn:network"],
)


py_library(
    name = "fqf",
    srcs = ["fqf.py"],
    deps = [":base_model",
    ],
)

py_library(
    name = "iqn",
    srcs = ["iqn.py"],
    deps = [":base_model",
    ],
)

py_library(
    name = "qrdqn",
    srcs = ["qrdqn.py"],
    deps = [":base_model",
    ],
)

