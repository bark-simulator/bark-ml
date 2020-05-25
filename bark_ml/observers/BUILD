py_library(
    name = "observers",
    srcs = ["nearest_state_observer.py",
            "simple_observer.py",
            "observer.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)

cc_library(
  name = "cpp_observers",
  hdrs = [
    "base_observer.hpp",
    "nearest_observer.hpp"
  ],
  deps = [
    "@bark_project//modules/world:world",
    "//bark_ml/commons:spaces",
    "//bark_ml/commons:commons"
  ],
  visibility = ["//visibility:public"],
)
