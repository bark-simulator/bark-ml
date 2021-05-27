load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


def _maybe(repo_rule, name, **kwargs):
  if name not in native.existing_rules():
    repo_rule(name = name, **kwargs)

def bark_ml_dependencies():
  _maybe(
    git_repository,
    name = "bark_project",
    branch = "pybind_bazel",
    remote = "https://github.com/bark-simulator/bark",
  )
  # _maybe(
  #   native.local_repository,
  #   name = "bark_project",
  #   path = "/Users/hart/Development/bark"
  # )

  _maybe(
      git_repository,
      name = "pybind11_bazel",
      commit="b16a4527a25cb82ba2e6bd9f831cbe89f5f50fd2",
      remote = "https://github.com/bark-simulator/pybind11_bazel"
  )

  # alternative to torch api used from virtual env
  _maybe(
    http_archive,
    name = "torch_api",
    urls = ["https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip"],
    build_file_content = """
cc_library(
    name = "lib",
    srcs = glob(["libtorch/lib/*.*"]),
    hdrs = glob(["libtorch/include/**/*.h", "libtorch/include/*.h"]),
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
    srcs = ["torch/lib/libc10.so", "torch/lib/libtorch_cpu.so"],
    hdrs = glob(["torch/include/**/*.h", "torch/include/*.h"]),
    visibility = ["//visibility:public"],
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
