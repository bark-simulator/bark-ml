py_library(
    name = "diadem",
    srcs = ["diadem_bark_environment.py"],
    data = ["@bark_project//python:bark.so"],
    imports=["diadem_project"],
    deps = [
        "@bark_project//modules/runtime:runtime",
        "@diadem_project//diadem:__init__"
        ],
    visibility = ["//visibility:public"],
)