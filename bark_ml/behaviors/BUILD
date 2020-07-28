py_library(
    name = "behaviors",
    srcs = glob(["*.py"]),
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
)