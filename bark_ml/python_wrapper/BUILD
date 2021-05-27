package(default_visibility = ["//visibility:public"])

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
  name = "core",
  srcs = glob(["*.cpp", "*.hpp"]),
  deps = [
    "//bark_ml/observers:cpp_observers",
    "//bark_ml/evaluators:cpp_evaluators",
    "//bark_ml/commons:spaces",
    "@com_github_eigen_eigen//:eigen",
  ],
  linkopts = ["-L/usr/local/lib/"]
)
