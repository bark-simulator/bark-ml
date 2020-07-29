py_library(
  name = "lib_tf2rl",
  deps = [
    ":tf2rl_wrapper",
    ":load_expert_trajectories",
    ":generate_expert_trajectories",
    ":compare_trajectories",
    "//bark_ml/library_wrappers/lib_tf2rl/agents",
    "//bark_ml/library_wrappers/lib_tf2rl/runners",
  ],
  visibility = ["//visibility:public"],
)

py_library(
    name = "tf2rl_wrapper",
    srcs = ["tf2rl_wrapper.py"],
    data = ["@bark_project//bark/python_wrapper:core.so"],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = ["@bark_project//bark/runtime:runtime",
    "@com_github_keiohta_tf2rl"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "load_expert_trajectories",
    srcs = [
      "load_expert_trajectories.py",
          ],
    deps = [
            "@com_github_keiohta_tf2rl",
            ":load_save_utils"
            ],
    visibility = ["//visibility:public"]
)

py_library(
  name = "load_save_utils",
  srcs = ["load_save_utils.py"],
  visibility = ["//visibility:public"]
)

py_library(
    name = "compare_trajectories",
    srcs = ["compare_trajectories.py"],
    deps = [":load_expert_trajectories"],
    visibility = ["//visibility:public"],
)


py_binary(
  name = "generate_expert_trajectories",
  srcs = [
    "generate_expert_trajectories.py",
    ],
  deps = ["//bark_ml/library_wrappers/lib_tf2rl:load_save_utils",
          "@bark_project//bark/runtime/commons:parameters",
          "@bark_project//bark/runtime/commons:xodr_parser",
          "@bark_project//bark/runtime/viewer:matplotlib_viewer",
          "@bark_project//bark/benchmark:benchmark_runner",
          "@bark_project//bark/benchmark:benchmark_analyzer",
          "@bark_project//bark/runtime/scenario/scenario_generation:scenario_generation",
          "@bark_project//bark/runtime:runtime",
          "//bark_ml/observers:observers"],
  data = [
          "@bark_project//bark/python_wrapper:core.so",
          "//bark_ml:generate_core"
          ],
  visibility = ["//visibility:public"]
)