
filegroup(
   name="configuration_data",
   srcs=glob(["*.json",
              "checkpoints/**"]),
   visibility = ["//visibility:public"],
)

py_library(
    name = "custom_evaluator",
    srcs = ["custom_evaluator.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "//src/evaluators:evaluators"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "custom_observer",
    srcs = ["custom_observer.py"],
    data = ["@bark_project//python:bark.so"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "//src/observers:observers"],
    visibility = ["//visibility:public"],
)

py_library(
  name = "configuration_lib",
  srcs = ["configuration_lib.py"],
  data = ["@bark_project//python:bark.so",
          "//tests/data:bark_data",
          ":configuration_data"],
  imports = ["../external/bark_project/python/"],
  deps = ["@bark_project//modules/runtime:runtime",
          "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
          "//src:rl_runtime",
          "//src/agents:agents",
          "//src/runners:runners",
          "//configurations:base_configuration",
          ":custom_evaluator",
          ":custom_observer",
          "@bark_project//modules/runtime/viewer:video_renderer"],
  visibility = ["//visibility:public"],
)

py_binary(
  name = "configuration",
  srcs = ["configuration.py"],
  data = ["@bark_project//python:bark.so",
          "//tests/data:bark_data",
          ":configuration_data"],
  imports = ["../external/bark_project/python/"],
  visibility = ["//visibility:public"],
  deps = [":configuration_lib"]
)