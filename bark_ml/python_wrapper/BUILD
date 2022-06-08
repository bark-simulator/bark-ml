package(default_visibility = ["//visibility:public"])


load("@pybind11_bazel//:build_defs.bzl", "pybind_library", "pybind_extension")

config_setting(
  name = "op_sys",
  define_values = { "build_platform": "macos" }
)

pybind_extension(
  name = "core",
  srcs = ["main.cpp", "pyobserver.hpp", "pynn_to_value_converter.hpp"],
  deps = [
    "//bark_ml/observers:cpp_observers",
    "//bark_ml/evaluators:cpp_evaluators",
    "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter:nn_to_value_converter",
    "//bark_ml/commons:spaces",
    "@bark_project//bark/commons/params:params",
    # external
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

pybind_library(
  name = "polymorphic_conversion",
  hdrs = [
    "polymorphic_conversion.hpp",
  ],
  srcs=[
    "polymorphic_conversion.cpp"
  ],
  deps = [
    "@bark_project//bark/python_wrapper:common",
    "//bark_ml/observers:cpp_observers",
    "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model/nn_to_value_converter:nn_to_value_converter",
  ],
  visibility = ["//visibility:public"],
)
