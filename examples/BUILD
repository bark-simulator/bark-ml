py_test(
  name = "continuous_env",
  srcs = ["continuous_env.py"],
  data = ["@bark_project//python:bark.so",
          "//python:bark_ml_library.so"],
  imports = ['../python/'],
  deps = [
    "//bark_ml/environments:gym",
  ]
)

py_test(
  name = "discrete_env",
  srcs = ["discrete_env.py"],
  data = ["@bark_project//python:bark.so",
          "//python:bark_ml_library.so"],
  imports = ['../python/'],
  deps = [
    "//bark_ml/environments:gym",
  ]
)

py_test(
  name = "adv_config",
  srcs = ["adv_config.py"],
  data = ["@bark_project//python:bark.so",
          "//python:bark_ml_library.so"],
  imports = ['../python/'],
  deps = [
    "//bark_ml/observers:observers",
    "//bark_ml/evaluators:evaluators",
    "//bark_ml/behaviors:behaviors",
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation"
  ]
)