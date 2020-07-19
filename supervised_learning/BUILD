py_test(
  name = "data_generation",
  srcs = ["data_generation.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ['../bark_ml/python_wrapper/'],
  deps = [
    "//bark_ml/environments:gym",
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "//bark_ml/library_wrappers/lib_tf2_gnn:lib_tf2_gnn",
    "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
    "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
  ],
  visibility = ["//visibility:public"],
)

py_test(
  name = "actor_nets",
  srcs = ["actor_nets.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ['../bark_ml/python_wrapper/'],
  deps = [
    "//bark_ml/environments:gym",
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "//bark_ml/library_wrappers/lib_tf2_gnn:lib_tf2_gnn",
    "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
  ],
  visibility = ["//visibility:public"],
)

py_test(
  name = "learner",
  srcs = ["learner.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ['../bark_ml/python_wrapper/'],
  deps = [
    ":actor_nets"
  ],
  visibility = ["//visibility:public"],
)

py_test(
  name = "data_handler",
  srcs = ["data_handler.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ['../bark_ml/python_wrapper/'],
  deps = [
    ":actor_nets",
    ":data_generation"
  ],
  visibility = ["//visibility:public"],
)

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
    ":data_generation",
    ":actor_nets",
    ":data_handler",
    ":learner"
  ]
)