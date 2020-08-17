test_suite(
  name = "unit_tests",
  tests = [
    "//bark_ml/tests:py_environment_tests",
    "//bark_ml/tests:py_observer_tests",
    "//bark_ml/tests:py_evaluator_tests",
    "//bark_ml/tests:py_behavior_tests",
    "//bark_ml/tests:py_library_tfa_tests"
  ]
)

test_suite(
  name = "graph_tests",
  tests = [
    "//bark_ml/tests:py_graph_observer_tests",
    "//bark_ml/tests:py_gnn_wrapper_tests",
    "//bark_ml/tests/capability_gnn_actor:py_gnn_actor_tests",
  ]
)

test_suite(
  name = "examples_tests",
  tests = [
    "//examples:blueprint_config",
    "//examples:continuous_env",
    "//examples:discrete_env",
    "//examples:tfa",
    "//examples:tfa_gnn"
  ]
)

test_suite(
  name = "diadem_tests",
  tests = [
    "//examples:diadem_dqn",
  ]
)
