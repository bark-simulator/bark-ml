package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "core.so",
  srcs = ["main.cpp", "pyobserver.hpp"],
  deps = [
    "//bark_ml/observers:cpp_observers",
    "//bark_ml/evaluators:cpp_evaluators",
    "//bark_ml/commons:spaces",
    "@bark_project//bark/commons/params:params",
    # external
    "@python_linux//:python-lib",
    "@pybind11//:pybind11",
    "@com_github_eigen_eigen//:eigen",
  ],
  linkshared = 1,
  linkstatic = 1
)

cc_library(
  name = "polymorphic_conversion",
  hdrs = [
    "polymorphic_conversion.hpp",
  ],
  srcs=[
    "polymorphic_conversion.cpp"
  ],
  deps = [
    "@bark_project//bark/python_wrapper:common",
    "@pybind11//:pybind11",
    "//bark_ml/observers:cpp_observers",
  ],
  visibility = ["//visibility:public"],
)
