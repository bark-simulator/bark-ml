


py_library(
    name = "evaluators",
    srcs = ["__init__.py",
            "general_evaluator.py",
            "evaluator_configs.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["@bark_project//bark/runtime:runtime"],
    visibility = ["//visibility:public"],
)

cc_library(
  name = "cpp_evaluators",
  hdrs = [
    "base_evaluator.hpp",
    "goal_reached.hpp"
  ],
  deps = [
    "@bark_project//bark/world:world",
    "@bark_project//bark/world/evaluation:evaluation",
    "//bark_ml/commons:spaces",
    "//bark_ml/commons:commons"
  ],
  visibility = ["//visibility:public"],
)