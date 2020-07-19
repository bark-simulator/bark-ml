filegroup(
    name = "notebooks_folder",
    srcs = glob(["*.ipynb"],
    exclude=["run.py", "run", "__init__.py"]),
    visibility = ["//visibility:public"],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = [
        "@bark_project//bark/python_wrapper:core.so",
        "//bark_ml:generate_core",
        "//examples/example_params:example_params",
        ":notebooks_folder",
        "//docs/report/data"
        ],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = [
        "//bark_ml/environments:single_agent_runtime",
        "//bark_ml/environments:blueprints",
    ]
)