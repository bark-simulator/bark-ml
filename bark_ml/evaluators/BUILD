py_library(
    name = "evaluators",
    srcs = ["goal_reached.py",
            "goal_reached_guiding.py",
            "evaluator.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)

cc_library(
  name = "cpp_evaluators",
  hdrs = [
    "base_evaluator.hpp",
    "goal_reached.hpp"
  ],
  deps = [
    "@bark_project//modules/world:world",
    "@bark_project//modules/world/evaluation:evaluation",
    "//bark_ml/commons:spaces",
    "//bark_ml/commons:commons"
  ],
  visibility = ["//visibility:public"],
)