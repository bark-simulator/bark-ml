py_library(
    name = "tfa_wrapper",
    srcs = ["tfa_wrapper.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../bark_ml/python_wrapper/"],
    deps = ["@bark_project//bark/runtime:runtime"],
    visibility = ["//visibility:public"],
)