py_library(
    name = "wrappers",
    srcs = ["action_wrapper.py",
						"tfa_wrapper.py"],
    data = ["@bark_project//python:bark.so",
            ":bark_scenarios"],
    imports = ["../external/bark_project/python/"],
		deps = ["@bark_project//modules/runtime:runtime"],
		visibility = ["//visibility:public"],
)