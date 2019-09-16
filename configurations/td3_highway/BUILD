
filegroup(
   name="configuration_data",
   srcs=glob(["*.json",
              "checkpoints/**"]),
   visibility = ["//visibility:public"],
)

py_binary(
  name = "configuration",
  srcs = ["configuration.py"],
  data = ["@bark_project//python:bark.so",
          "//tests/data:bark_data",
          ":configuration_data"],
  imports = ["../external/bark_project/python/"],
  deps = ["@bark_project//modules/runtime:runtime",
          "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
          "//src:rl_runtime",
          "//src/agents:agents",
          "//src/runners:runners",
          "//configurations:base_configuration"],
  visibility = ["//visibility:public"],
)