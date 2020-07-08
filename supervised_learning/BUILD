

py_test(
  name = "py_learn_actor_tests",
  srcs = ["py_learn_actor_tests.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ['../python/'],
  deps =   [
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "//bark_ml/library_wrappers/lib_tf2_gnn:lib_tf2_gnn",
    "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
    "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
    "//examples:data_generation"
  ]
)