py_library(
    name = "tf2rl_wrapper",
    srcs = ["tf2rl_wrapper.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "load_expert_trajectories",
    srcs = ["load_expert_trajectories.py"],
    visibility = ["//visibility:public"]
)

py_binary(
  name = "generate_expert_trajectories",
  srcs = ["generate_expert_trajectories.py"],
  deps = ["@bark_project//modules/runtime/commons:parameters",
          "@bark_project//modules/runtime/commons:xodr_parser",
          "@bark_project//modules/runtime/viewer:matplotlib_viewer",
          "@bark_project//modules/benchmark:benchmark_runner",
          "@bark_project//modules/benchmark:benchmark_analyzer",
          "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
          "@bark_project//modules/runtime:runtime",
          "//bark_ml/observers:observers"],
  data = ["@bark_project//python:bark.so",
          "@bark_project//modules/runtime/tests:xodr_data",
          "//python:bark_ml_library.so",
          ],
  visibility = ["//visibility:public"]
)