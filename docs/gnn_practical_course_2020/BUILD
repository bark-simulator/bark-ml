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
        "//docs/gnn_practical_course_2020/data",
        "//docs/gnn_practical_course_2020/images",
        "//bark_ml/tests/capability_gnn_actor/data:data"
        ],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = [
        "//bark_ml/environments:single_agent_runtime",
        "//bark_ml/environments:blueprints",
        "//bark_ml/observers",
        "//bark_ml/library_wrappers/lib_tf_agents/networks:networks",
        "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
        "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
        "//bark_ml/tests/capability_gnn_actor:data_generation",
        "//bark_ml/tests/capability_gnn_actor:data_handler",
        "//bark_ml/tests/capability_gnn_actor:actor_nets",
        "//bark_ml/tests/capability_gnn_actor:learner",
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
    "//bark_ml/library_wrappers/lib_tf_agents/networks:networks",
    "//bark_ml/library_wrappers/lib_tf_agents/agents:agents",
    "//bark_ml/library_wrappers/lib_tf_agents/runners:runners",
  ],
  visibility = ["//visibility:public"],
)