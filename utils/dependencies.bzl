load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def _maybe(repo_rule, name, **kwargs):
  # if name not in native.existing_rules():
  repo_rule(name = name, **kwargs)

def bark_ml_dependencies():
  _maybe(
    git_repository,
    name = "bark_project",
    branch = "master",
    remote = "https://github.com/bark-simulator/bark",
  )
  # _maybe(
  #   native.local_repository,
  #   name = "bark_project",
  #   path = "/Users/hart/Development/bark"
  # )
  _maybe(
    native.new_local_repository,
    name = "python_linux",
    path = "./bark_ml/python_wrapper/venv/",
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
    commit = "741b9ea7a96657e399ae039ab922a8baf0b0fce1",
    remote = "https://github.com/juloberno/diadem"
  )

  _maybe(
    git_repository,
    name = "libtensorflow-RL-MCTS",
    commit = "0573ec5ca275da195802ab73a69e87947309ac03",
    remote = "https://github.com/steven-guo94/libtensorflow_so"
  )

  ## this repository named "libtensorflow-RL-MCTS" is a precompiled tensorflow2.1 C API
  ## it is build according to this tutorial https://medium.com/analytics-vidhya/deploying-tensorflow-2-1-as-c-c-executable-1d090845055c

  # if we include glog twice, gflags are defined mult. times
  _maybe(
    new_git_repository,
    name = "com_github_google_glog",
    commit = "195d416e3b1c8dc06980439f6acd3ebd40b6b820",
    remote = "https://github.com/google/glog",
    build_file="//:utils/glog.BUILD"
  )