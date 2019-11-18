py_test(
    name = "py_bark_agent_tests",
    srcs = ["py_bark_agent_tests.py"],
    data = ["@bark_project//python:bark.so",
            "//tests/data:bark_data",
            "//configurations/sac_highway:configuration_data"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//configurations/sac_highway:configuration_lib",
            "//configurations:bark_agent",
            "//src/agents:agents"],
    visibility = ["//visibility:public"],
)


py_test(
    name = "py_runtime_rl_tests",
    srcs = ["py_runtime_rl_tests.py"],
    data = ["@bark_project//python:bark.so",
            "//tests/data:bark_data"],
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
            "//tests/data:bark_data"],
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
            "//tests/data:bark_data"],
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
            "//tests/data:bark_data"],
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
          "//tests/data:bark_data"],
  imports = ["../external/bark_project/python/"],
  deps = ["@bark_project//modules/runtime:runtime",
          "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation"],
)
