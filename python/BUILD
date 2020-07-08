package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "bark_ml_library.so",
  srcs = glob(["*.cpp", "*.hpp"]),
  deps = [
    "//bark_ml/observers:cpp_observers",
    "//bark_ml/evaluators:cpp_evaluators",
    "//bark_ml/commons:spaces",
    # external
    "@python_linux//:python-lib",
    "@pybind11//:pybind11",
    "@com_github_eigen_eigen//:eigen",
  ],
  linkshared = 1,
  linkstatic = 1
)
