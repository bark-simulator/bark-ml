load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)

def bark_ml_dependencies():
  _maybe(
    git_repository,
    name = "bark_project",
    commit = "6e1080ed07f9a47973e553c78697910c90b84bf2",
    remote = "https://github.com/bark-simulator/bark",
  )

  _maybe(
    native.new_local_repository,
    name = "python_linux",
    path = "./python/venv/",
    build_file_content = """
cc_library(
    name = "python-lib",
    srcs = glob(["lib/libpython3.*", "libs/python3.lib", "libs/python36.lib"]),
    hdrs = glob(["include/**/*.h", "include/*.h"]),
    includes = ["include/python3.6m", "include", "include/python3.7m", "include/python3.5m"], 
    visibility = ["//visibility:public"],
)
    """)

  _maybe(
    git_repository,
    name = "diadem_project",
    commit = "e1d5cef06499e3eecfe96c774958237321933dfd",
    remote = "https://github.com/juloberno/diadem"
  )