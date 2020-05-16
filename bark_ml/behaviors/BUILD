py_library(
    name = "behaviors",
    srcs = glob(["*.py"]),
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
)