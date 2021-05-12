load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


def _maybe(repo_rule, name, **kwargs):
  if name not in native.existing_rules():
    repo_rule(name = name, **kwargs)

def bark_ml_dependencies():
  _maybe(
    git_repository,
    name = "bark_project",
    commit = "6d390caf755c65a027b6b3cbd2d6c952e81765f6",
    remote = "https://github.com/bark-simulator/bark",
  )
 # _maybe(
 #   native.local_repository,
 #   name = "bark_project",
 #   path = "/home/julo/development/bark"
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
    native.new_local_repository,
    name = "torchcpp",
    path = "./bark_ml/python_wrapper/venv/lib/python3.7/site-packages/",
    build_file_content = """
cc_library(
    name = "lib",
    srcs = ["torch/lib/libtorch.so",
                 "torch/lib/libc10.so", "torch/lib/libtorch_cpu.so"],
    hdrs = glob(["torch/include/**/*.h", "torch/include/*.h"]),
    visibility = ["//visibility:public"],
    linkopts = [
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
    ],
)
    """)
  _maybe(
    git_repository,
    name = "diadem_project",
    commit = "64b2987fbdd69ad533f30b545568c691ad5afb00",
    remote = "https://github.com/juloberno/diadem"
  )

  # if we include glog twice, gflags are defined mult. times
  _maybe(
    new_git_repository,
    name = "com_github_google_glog",
    commit = "195d416e3b1c8dc06980439f6acd3ebd40b6b820",
    remote = "https://github.com/google/glog",
    build_file="//:utils/glog.BUILD"
  )
