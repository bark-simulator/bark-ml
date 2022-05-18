load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
  name = "pytorch_script_wrapper",
  srcs = ["pytorch_script_wrapper.cpp"],
  deps = [":model_loader",
          "@torchcpp//:lib",
          ],
  copts = ["-Iexternal/torchcpp/torch/include",
           "-Lexternal/torchcpp/torch/lib",
           "-D_GLIBCXX_USE_CXX11_ABI=0"],
   linkopts = ["-ltorch_cpu","-lc10", "-ltorch"],
   linkshared = 1,
   linkstatic = 1,
   visibility = ["//visibility:public"],
)

cc_library(
  name = "model_loader",
  hdrs = ["model_loader.hpp"],
  deps = ["@torchcpp//:lib"],
  linkstatic=False,
  visibility = ["//visibility:public"],
)