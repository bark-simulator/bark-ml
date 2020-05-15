py_test(
    name = "py_environment_tests",
    srcs = ["py_environment_tests.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
            "//bark_ml/environments:single_agent_runtime",
            "//bark_ml/commons:py_spaces"],
    visibility = ["//visibility:public"],
)

cc_binary(
  name = "observer_tests",
  srcs = [
    "observer_tests.cc",
  ],
  copts = ["-Iexternal/gtest/include"],
  deps = [
    "//bark_ml/observers:nearest_observer",
    "@bark_project//modules/world/tests:make_test_world",
    "@gtest//:main",
  ]
)