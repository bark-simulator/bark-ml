py_library(
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
    "//bark_ml/library_wrappers/lib_tf_agents/networks:networks",
    "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
    "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
  ],
  visibility = ["//visibility:public"],
)

py_library(
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
    "//bark_ml/library_wrappers/lib_tf_agents/networks:networks",
    "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
  ],
  visibility = ["//visibility:public"],
)

py_library(
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

py_library(
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
  name = "py_gnn_actor_tests",
  srcs = ["py_gnn_actor_tests.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params",
          "//bark_ml/tests/capability_gnn_actor/data:data"],
  imports = ['../python/'],
  deps =   [
    "//bark_ml/environments:single_agent_runtime",
    "//bark_ml/environments:blueprints",
    "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
    "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
    ":data_generation",
    ":actor_nets",
    ":data_handler",
    ":learner"
  ]
)