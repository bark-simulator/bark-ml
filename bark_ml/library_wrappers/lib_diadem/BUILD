py_library(
    name = "diadem",
    srcs = ["diadem_bark_environment.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports=["diadem_project"],
    deps = [
        "@bark_project//bark/runtime:runtime",
        "@diadem_project//diadem:__init__"
        ],
    visibility = ["//visibility:public"],
)