load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


def _maybe(repo_rule, name, **kwargs):
  if name not in native.existing_rules():
    repo_rule(name = name, **kwargs)

def bark_ml_dependencies():
  _maybe(
    git_repository,
    name = "bark_project",
    branch = "master",
    remote = "https://github.com/bark-simulator/bark",
  )

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
)"""
  )

  _maybe(
      git_repository,
      name = "pybind11_bazel",
      commit="c4a29062b77bf42836d995f6ce802f642cffb939",
      remote = "https://github.com/bark-simulator/pybind11_bazel"
  )
