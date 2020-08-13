py_library(
    name = "behaviors",
    srcs = glob(["*.py"]),
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../bark_ml/python_wrapper/"],
    deps = ["//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
)