py_test(
    name = "base_agent_test",
    srcs = ["base_agent_test.py"],
    data = [
            "@bark_project//bark/python_wrapper:core.so"
            ],
    imports = ["../external/bark_project/bark/python_wrapper/",
              ],
    deps = ["//bark_ml/environments:single_agent_runtime",
            "//bark_ml/environments:gym",
            "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
            "//bark_ml/commons:py_spaces"
           ],
    visibility = ["//visibility:public"],
)
