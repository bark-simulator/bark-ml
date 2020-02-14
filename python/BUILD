package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "bark_ml.so",
  srcs = glob(["*.cpp", "*.hpp"]),
  deps = [
    "//modules/runtime/tests:py_param_server_test_helper",
    "//modules/geometry:geometry",
    "//modules/world/opendrive:opendrive",
    "//modules/world/tests:make_test_xodr_map",
    "//modules/world/tests:make_test_world",
    "//modules/world/map:roadgraph",
    "//modules/world/goal_definition:goal_definition",
    "//modules/world/evaluation:evaluation",
    "//modules/commons/params:params",
    "//modules/commons/util:util",
    "//modules/world:world",
    # external
    "@python_linux//:python-lib",
    "@pybind11//:pybind11",
    "@com_github_eigen_eigen//:eigen",
  ],
  linkshared = 1,
  linkstatic = 1
)
