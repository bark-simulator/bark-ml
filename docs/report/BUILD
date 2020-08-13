filegroup(
    name = "notebooks_folder",
    srcs = glob(["*.ipynb"],
    exclude=["run.py", "run", "__init__.py"]),
    visibility = ["//visibility:public"],
)

# add bark deps here
py_binary(
    name = "run",
    srcs = ["run.py"],
    data = [
        "@bark_project//bark/python_wrapper:core.so",
        "//bark_ml:generate_core",
        ":notebooks_folder",
        "//docs/report/data",
        "//docs/report/data:data_dirs",
        "@com_github_gail_4_bark_large_data_store"
        ],
    imports = ["../external/bark_project/bark/python_wrapper/"],
    deps = [
        "//bark_ml/environments:single_agent_runtime",
        "//bark_ml/environments:blueprints",
        "//bark_ml/library_wrappers/lib_tf2rl",
    ]
)