py_library(
    name = "evaluator_stl",
    srcs = ["__init__.py",
            "evaluator_stl.py",
            "label_functions/base_label_function.py",
            "label_functions/safe_distance_label_function.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../../external/bark_project/bark/python_wrapper/"],
    visibility = ["//visibility:public"],
)