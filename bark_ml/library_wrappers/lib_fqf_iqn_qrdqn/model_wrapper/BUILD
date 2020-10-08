cc_binary(
  name = "pytorch_script_wrapper.so",
  srcs = ["pytorch_script_wrapper.cpp"],
  deps = [":model_loader",
          "@torchcpp//:lib",
          "@python_linux//:python-lib",
          "@pybind11//:pybind11",
          ],
  copts = ["-Iexternal/torchcpp/torch/include",
           "-Lexternal/torchcpp/torch/lib",
           "-D_GLIBCXX_USE_CXX11_ABI=0"],

   linkopts = ["-ltorch_cpu","-lc10"],
   linkshared = 1,
   linkstatic = 1,
   visibility = ["//visibility:public"],
)

cc_library(
  name = "model_loader",
  hdrs = ["model_loader.hpp"],
  deps = ["@torchcpp//:lib"],
  copts = ["-Iexternal/torchcpp/torch/include",
           "-Lexternal/torchcpp/torch/lib",
           "-D_GLIBCXX_USE_CXX11_ABI=0"],

  linkopts = ["-ltorch_cpu","-lc10"],
  linkstatic=False,
)