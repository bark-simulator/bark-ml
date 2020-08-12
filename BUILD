test_suite(
  name = "all_tests",
  tests = [
    ":unit_tests",
    ":gail_tests",
    ":generate_load_tests"
  ]
)

test_suite(
  name = "unit_tests",
  tests = [
    "//bark_ml/tests:py_environment_tests",
    "//bark_ml/tests:py_observer_tests",
    "//bark_ml/tests:py_evaluator_tests",
    "//bark_ml/tests:py_behavior_tests",
    "//utils/tests:test_generate_launch_configuration",
  ]
)

test_suite(
  name = "gail_tests",
  tests = [
    "//bark_ml/tests/py_library_tf2rl_tests:normalization_utils_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:py_gail_agent_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:py_tf2rl_wrapper_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:py_gail_runner_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:py_gail_training_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:compare_trajectories_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:normalize_on_merging_blueprint_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:calculate_action_tests",
    ]
)

test_suite(
  name = "generate_load_tests",
  tests = [
    "//bark_ml/tests/py_library_tf2rl_tests:base_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:simulation_based_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:load_save_utils_tests",
    "//bark_ml/tests/py_library_tf2rl_tests:load_expert_trajectories_tests",
  ]
)

test_suite(
  name = "examples_tests",
  tests = [
    "//examples:blueprint_config",
    "//examples:continuous_env",
    "//examples:discrete_env",
    "//examples:tfa",
    "//examples:tfa_generate",
    "//examples:gail",
  ]
)

test_suite(
  name = "diadem_tests",
  tests = [
    "//examples:diadem_dqn",
  ]
)
