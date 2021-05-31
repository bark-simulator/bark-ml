package(default_visibility = ["//visibility:public"])

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

config_setting(
    name = "op_sys",
    define_values = { "build_plattform": "macos" }
)

pybind_extension(
  name = "core",
  srcs = glob(["*.cpp", "*.hpp"]),
  deps = [
    "//bark_ml/observers:cpp_observers",
    "//bark_ml/evaluators:cpp_evaluators",
    "//bark_ml/commons:spaces",
    "@com_github_eigen_eigen//:eigen",
  ],
  linkopts = select({
    ":op_sys": [""],
    "//conditions:default": ["-L/usr/local/lib/ -Wl,-Bsymbolic"],
  }),
  linkstatic = select({
    ":op_sys": 0,
    "//conditions:default": 1,
  })
)
