genrule(
  name = "generate_core",
  srcs = [
    "//bark_ml/python_wrapper:core.so",
  ],
  cmd = ("mv $(RULEDIR)/python_wrapper/core.so  $(location core.so)"),
  outs = [
    "core.so",
  ],
  visibility = ["//visibility:public"],
)

py_test(
  name = "pip_package",
  srcs = ["pip_package.py", "__init__.py"],
  data = [":generate_core"],
  deps = []
)

test_suite(
  name = "unit_tests",
  tests = [
    "//bark_ml/tests:py_environment_tests",
    "//bark_ml/tests:py_observer_tests",
    "//bark_ml/tests:py_graph_observer_tests",
    "//bark_ml/tests:py_evaluator_tests",
    "//bark_ml/tests:py_behavior_tests",
    "//bark_ml/tests:py_library_tfa_tests",
    "//bark_ml/tests:py_bark_behavior_model_tests",
    "//bark_ml/tests:py_tracer_tests",
    "//bark_ml/experiment:py_experiment_tests"
  ]
)

test_suite(
  name = "examples_tests",
  tests = [
    "//examples:blueprint_config",
    "//examples:continuous_env",
    "//examples:discrete_env",
    "//examples:tfa",
    "//examples:tfa_gnn",
    "//examples:tfa_counterfactual"
  ]
)