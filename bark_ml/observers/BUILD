py_library(
    name = "observers",
    srcs = ["__init__.py",
            "nearest_state_observer.py",
            "simple_observer.py",
            "observer.py",
            "graph_observer.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["@bark_project//bark/runtime:runtime"],
    visibility = ["//visibility:public"],
)

cc_library(
  name = "cpp_observers",
  hdrs = [
    "base_observer.hpp",
    "nearest_observer.hpp",
    "frenet_observer.hpp",
    "static_observer.hpp"
  ],
  deps = [
    "@bark_project//bark/world:world",
    "//bark_ml/commons:spaces",
    "//bark_ml/commons:commons"
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "include",
  hdrs = [
    "base_observer.hpp",
    "nearest_observer.hpp",
  ],
  deps=[
    "//bark_ml/commons:spaces",
    "//bark_ml/commons:commons"
  ],
  visibility = ["//visibility:public"],
)
