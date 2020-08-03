


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
    "//bark_ml/environments:gym",
  ]
)

py_test(
  name = "fqf",
  srcs = ["fqf.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ["../bark_ml/python_wrapper/"],
  deps = [
    "//bark_ml/environments:gym",
    "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
  ]
)

cc_test(
  name = "pytorch_model_cpp",
  srcs = ["pytorch_model_cpp.cc"],
  deps = ["@torchcpp//:lib",],
  copts = ["-Iexternal/torchcpp/torch/include",
           #"-Lexternal/torchcpp/libtorch/lib",
           "-D_GLIBCXX_USE_CXX11_ABI=0"],

  linkopts = ["-ltorch","-lc10"],
  linkstatic=False,
)

cc_test(
  name = "pytorch_model_cpp_observer",
  srcs = ["pytorch_model_cpp_observer.cc"],
  deps = ["@torchcpp//:lib",
          "//bark_ml/observers:cpp_observers",
          "@bark_project//bark/geometry",
          "@bark_project//bark/commons:commons",
          "@bark_project//bark/world:world",
          "@bark_project//bark/world/map:map_interface",
          "@bark_project//bark/world/map:roadgraph",
          "@bark_project//bark/world/opendrive:opendrive",
          "@bark_project//bark/models/behavior/constant_velocity:constant_velocity",
          "@bark_project//bark/models/behavior/motion_primitives:motion_primitives",
          "@bark_project//bark/models/execution/interpolation:interpolation",
          "@bark_project//bark/world/tests:make_test_world",
          "@bark_project//bark/world/tests:dummy_road_corridor",
          ],
  copts = ["-Iexternal/torchcpp/torch/include",
           #"-Lexternal/torchcpp/libtorch/lib",
           "-D_GLIBCXX_USE_CXX11_ABI=0"],

  linkopts = ["-ltorch","-lc10"],
  linkstatic=False,
)

py_test(
  name = "iqn",
  srcs = ["iqn.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ["../bark_ml/python_wrapper/"],
  deps = [
    "//bark_ml/environments:gym",
    "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
  ]
)

py_test(
  name = "qrdqn",
  srcs = ["qrdqn.py"],
  data = ["@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core",
          "//examples/example_params:example_params"],
  imports = ["../bark_ml/python_wrapper/"],
  deps = [
    "//bark_ml/environments:gym",
    "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
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
