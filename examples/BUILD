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
  name = "blueprint_config",
  srcs = ["blueprint_config.py"],
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

py_test(
  name = "tfa",
  srcs = ["tfa.py"],
  data = ["@bark_project//python:bark.so",
          "//python:bark_ml_library.so",
          "//examples/example_params:example_params"],
  imports = ['../python/'],
  deps = [
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
    "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
  ]
)

py_test(
  name = "diadem_dqn",
  srcs = ["diadem_dqn.py"],
  data = ["@bark_project//python:bark.so",
          "//python:bark_ml_library.so",
          "//examples/example_params:example_params"
        ],
  imports = ['../python/', "./external/diadem_project/"],
  deps = [
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "//bark_ml/observers:observers",
    "//bark_ml/library_wrappers/lib_diadem:diadem"
  ]
)