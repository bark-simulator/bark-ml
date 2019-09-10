py_library(
    name = "wrappers",
    srcs = ["action_wrapper.py",
						"tfa_wrapper.py",
            "dynamic_model.py",
            "motion_primitives.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
		deps = ["@bark_project//modules/runtime:runtime",
            "//src/commons:spaces"],
		visibility = ["//visibility:public"],
)