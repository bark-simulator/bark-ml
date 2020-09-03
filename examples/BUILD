


py_test(
  name = "continuous_env",
  srcs = ["continuous_env.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core"],
  imports = ["../bark_ml/python_wrapper/"],
  deps = [
    "//bark_ml/environments:gym",
  ]
)

py_test(
  name = "discrete_env",
  srcs = ["discrete_env.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core"],
  imports = ["../bark_ml/python_wrapper/"],
  deps = [
    "//bark_ml/environments:gym",
  ]
)

py_test(
  name = "blueprint_config",
  srcs = ["blueprint_config.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core"],
  imports = ["../bark_ml/python_wrapper/"],
  deps = [
    "//bark_ml/observers:observers",
    "//bark_ml/evaluators:evaluators",
    "//bark_ml/behaviors:behaviors",
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "@bark_project//bark/runtime/scenario/scenario_generation:scenario_generation"
  ]
)

py_test(
  name = "tfa",
  srcs = ["tfa.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ["../bark_ml/python_wrapper/"],
  deps = [
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
    "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
  ]
)

py_test(
  name = "tfa_gnn",
  srcs = ["tfa_gnn.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ["../bark_ml/python_wrapper/"],
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
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"
        ],
  imports = ["../bark_ml/python_wrapper/",
             "./external/diadem_project/"],
  deps = [
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "//bark_ml/observers:observers",
    "//bark_ml/library_wrappers/lib_diadem:diadem"
  ]
)
