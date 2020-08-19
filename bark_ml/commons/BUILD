cc_library(
  name="spaces",
  hdrs = [
    "spaces.hpp",
  ],
  deps = [
    "@com_github_eigen_eigen//:eigen",
  ],
  visibility = ["//visibility:public"],
)

cc_library(
  name="commons",
  hdrs = [
    "commons.hpp",
  ],
  deps = [
    "@com_github_eigen_eigen//:eigen",
  ],
  visibility = ["//visibility:public"],
)

py_library(
    name = "py_spaces",
    srcs = ["py_spaces.py"],
    visibility = ["//visibility:public"],
)


py_library(
  name = "tracer",
  srcs = ["tracer.py"],
  visibility = ["//visibility:public"],
)