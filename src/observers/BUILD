py_library(
    name = "observers",
    srcs = ["nearest_state_observer.py",
            "simple_observer.py",
            "observer.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "//src/commons:spaces"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nearest_observer",
    hdrs = [
        "nearest_observer.hpp",
    ],
    deps = [
      "@bark_project//modules/commons/params:params",
      "@bark_project//modules/world:world",
    ],
    visibility = ["//visibility:public"],
)
