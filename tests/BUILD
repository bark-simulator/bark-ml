
filegroup(
   name="bark_scenarios",
   srcs=glob(["data/**"]),
   visibility = ["//visibility:public"],
)


py_test(
    name = "py_bark_test",
    srcs = ["py_bark_test.py"],
    data = ["@bark_project//python:bark.so",
            ":bark_scenarios"],
    deps = ["@bark_project//modules/runtime:runtime"],
    imports = ["../external/bark_project/python/"],
    visibility = ["//visibility:public"]
)

cc_test(
    name = "cpp_bark_test",
    srcs = [
        "cpp_bark_test.cc",
    ],
    copts = ["-Iexternal/gtest/include"],
    deps = [
        "@bark_project//modules/geometry",
        "@bark_project//modules/models/dynamic:dynamic",
        "@gtest//:main",
        "@boost//:geometry"
    ]
)


# py_test(
#     name = "py_runtime_rl_tests",
#     srcs = ["py_runtime_rl_tests.py"],
#     data = ['//python:bark.so',
#             '//modules/runtime/tests:xodr_data'],
#     imports = ['../../../python'],
#     deps = ["//modules/runtime/ml:ml",
#             "//modules/runtime/commons:parameters",
#             "//modules/runtime/commons:commons",
#             "//modules/runtime/scenario/scenario_generation:scenario_generation",
#             "//modules/runtime/commons:xodr_parser"],
#     visibility = ["//visibility:public"],
# )

# py_test(
#     name = "py_runtime_wrapper_tests",
#     srcs = ["py_runtime_wrapper_tests.py"],
#     data = ['//python:bark.so',
#             '//modules/runtime/tests:xodr_data'],
#     imports = ['../../../python'],
#     deps = ["//modules/runtime/ml:ml",
#             "//modules/runtime/commons:parameters",
#             "//modules/runtime/commons:commons",
#             "//modules/runtime/scenario/scenario_generation:scenario_generation",
#             "//modules/runtime/commons:xodr_parser"],
#     visibility = ["//visibility:public"],
# )


# py_test(
#     name = "py_runtime_tfa_tests",
#     srcs = ["py_runtime_tfa_tests.py"],
#     data = ['//python:bark.so',
#             '//modules/runtime/tests:xodr_data'],
#     imports = ['../../../python'],
#     deps = ["//modules/runtime/ml:ml",
#             "//modules/runtime/commons:parameters",
#             "//modules/runtime/commons:commons",
#             "//modules/runtime/scenario/scenario_generation:scenario_generation",
#             "//modules/runtime/commons:xodr_parser"],
#     visibility = ["//visibility:public"],
# )

py_test(
    name = "py_agent_tests",
    srcs = ["py_agent_tests.py"],
    data = ["@bark_project//python:bark.so",
            ":bark_scenarios"],
    imports = ["../external/bark_project/python/"],
    deps = ["@bark_project//modules/runtime:runtime",
            "@bark_project//modules/runtime/scenario/scenario_generation:scenario_generation",
            "//src:rl_runtime",
            "//src/agents:agents"],
    visibility = ["//visibility:public"],
)


# py_test(
#     name = "py_runner_tests",
#     srcs = ["py_runner_tests.py"],
#     imports = ['../../../python'],
#     data = ['//python:bark.so',
#             '//modules/runtime/tests:xodr_data'],
#     deps = ["//modules/runtime/ml:ml",
#             "//modules/runtime/commons:parameters",
#             "//modules/runtime/commons:commons",
#             "//modules/runtime/scenario/scenario_generation:scenario_generation",
#             "//modules/runtime/commons:xodr_parser"],
#     visibility = ["//visibility:public"],
# )
