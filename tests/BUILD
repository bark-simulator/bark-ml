py_test(
    name = "py_runtime_wrapper_tests",
    srcs = ["py_runtime_wrapper_tests.py"],
    data = ["@bark_project//python:bark.so",
            "//data:bark_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_runtime_rl_tests",
    srcs = ["py_runtime_rl_tests.py"],
    data = ["@bark_project//python:bark.so",
            "//data:bark_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_runtime_tfa_tests",
    srcs = ["py_runtime_tfa_tests.py"],
    data = ["@bark_project//python:bark.so",
            "//data:bark_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_agent_tests",
    srcs = ["py_agent_tests.py"],
    data = ["@bark_project//python:bark.so",
            "//data:bark_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "py_runner_tests",
    srcs = ["py_runner_tests.py"],
    data = ["@bark_project//python:bark.so",
            "//data:bark_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents",
            "//src/runners:runners"],
    visibility = ["//visibility:public"],
)

py_test(
  name = "py_runtime_tests",
  srcs = ["py_runtime_tests.py"],
  data = ["@bark_project//python:bark.so",
          "//data:bark_data"],
  imports = ["../external/bark_project/python/"],
  deps = ["@bark_project//modules/runtime:runtime",
          "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation"],
)
