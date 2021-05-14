genrule(
  name = "generate_core",
  srcs = [
    "//bark_ml/python_wrapper:core.so",
  ],
  cmd = (" mv $(RULEDIR)/python_wrapper/core.so  $(location core.so)"),
  outs = [
    "core.so",
  ],
  visibility = ["//visibility:public"],
)

py_test (
    name="py_tests",
    data = [':generate_core'],
    srcs = ["py_tests.py",
            "//bark_ml/tests:py_behavior_tests",
            "//bark_ml/tests:py_environment_tests",
            "//bark_ml/tests:py_evaluator_tests",
            "//bark_ml/tests:py_library_tfa_tests",
            "//bark_ml/tests:py_observer_tests"
    ],
)

py_test(
  name = "pip_package",
  srcs = ["pip_package.py", "__init__.py"],
  data = [":generate_core", "//:readme"],
  deps = [":py_tests"]
)