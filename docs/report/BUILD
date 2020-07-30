filegroup(
    name = "notebooks_folder",
    srcs = glob(["*.ipynb"],
    exclude=["run.py", "run", "__init__.py"]),
    visibility = ["//visibility:public"],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = [
        "@bark_project//bark/python_wrapper:core.so",
        "//bark_ml:generate_core",
        "//examples/example_params:example_params",
        ":notebooks_folder",
        "//docs/report/data"
        ],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = [
        "//bark_ml/environments:single_agent_runtime",
        "//bark_ml/environments:blueprints",
        "//bark_ml/observers:graph_observer",
        "//bark_ml/library_wrappers/lib_tf2_gnn:lib_tf2_gnn",
        "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
        "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
        "//supervised_learning:data_generation",
        "//supervised_learning:data_handler",
        "//supervised_learning:actor_nets",
        "//supervised_learning:learner",
        ":helper_functions"
    ]
)

py_library(
  name = "helper_functions",
  srcs = ["helper_functions.py"],
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