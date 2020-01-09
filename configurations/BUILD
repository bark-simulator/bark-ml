py_library(
    name = "bark_agent",
    srcs = ["bark_agent.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "//configurations/sac_highway:configuration"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "base_configuration",
    srcs = ["base_configuration.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)