py_library(
    name = "wrappers",
    srcs = ["action_wrapper.py",
						"tfa_wrapper.py",
            "dynamic_model.py",
            "motion_primitives.py"],
    data = ["@bark_project//python:bark.so",
            "//python:bark_ml.so",
            "//src/commons:py_spaces"],
    imports = ["../external/bark_project/python/",
               "../../python/"],
		deps = ["@bark_project//modules/runtime:runtime"],
		visibility = ["//visibility:public"],
)