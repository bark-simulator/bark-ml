cc_binary(
  name = "pytorch_script_wrapper.so",
  srcs = ["pytorch_script_wrapper.cpp"],
  data=["//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model_wrapper/model_data:weights"],
  deps = [":model_loader",
          "@torchcpp//:lib",
          "@python_linux//:python-lib",
          "@pybind11//:pybind11",
          ],
  copts = ["-Iexternal/torchcpp/torch/include",
           "-Lexternal/torchcpp/libtorch/lib",
           "-D_GLIBCXX_USE_CXX11_ABI=0"],

   linkopts = ["-ltorch_cpu","-lc10"],
   linkshared = 1,
   linkstatic = 1,
   visibility = ["//visibility:public"],
)

cc_test(
  name = "pytorch_model_cpp",
  srcs = ["pytorch_model_cpp.cc"],
  data=["//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model_wrapper/model_data:weights"],
  deps = [":model_loader",
          "@torchcpp//:lib",
          "@gtest//:gtest"],

  copts = ["-Iexternal/torchcpp/torch/include",
           #"-Lexternal/torchcpp/torch/lib",
           "-Iexternal/gtest/include",
           "-D_GLIBCXX_USE_CXX11_ABI=0"],

  linkopts = ["-ltorch_cpu","-lc10"],
  linkstatic=False
)

cc_library(
  name = "model_loader",
  hdrs = ["model_loader.hpp"],
  deps = ["@torchcpp//:lib"],
  copts = ["-Iexternal/torchcpp/torch/include",
           "-D_GLIBCXX_USE_CXX11_ABI=0"],

  linkopts = ["-ltorch_cpu","-lc10"],
  linkstatic=False,
)

cc_test(
  name = "pytorch_model_cpp_observer",
  srcs = ["pytorch_model_cpp_observer.cc"],
  deps = ["@torchcpp//:lib",
          "//bark_ml/observers:cpp_observers",
          "@bark_project//bark/geometry",
          "@bark_project//bark/commons:commons",
          "@bark_project//bark/world:world",
          "@bark_project//bark/world/map:map_interface",
          "@bark_project//bark/world/map:roadgraph",
          "@bark_project//bark/world/opendrive:opendrive",
          "@bark_project//bark/models/behavior/constant_velocity:constant_velocity",
          "@bark_project//bark/models/behavior/motion_primitives:motion_primitives",
          "@bark_project//bark/models/execution/interpolation:interpolation",
          "@bark_project//bark/world/tests:make_test_world",
          "@bark_project//bark/world/tests:dummy_road_corridor",
          ],
  copts = ["-Iexternal/torchcpp/torch/include",
           #"-Lexternal/torchcpp/libtorch/lib",
           "-D_GLIBCXX_USE_CXX11_ABI=0"],

  linkopts = ["-ltorch_cpu","-lc10"],
  linkstatic=False,
)

py_test(
  name = "torch_script",
  srcs = ["torch_script.py"],
  data = ["@torchcpp//:lib",
          "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model_wrapper/model_data:weights",
          "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/model_wrapper:pytorch_script_wrapper.so",
          "@bark_project//bark/runtime/commons:commons"],
  imports = ["external/torchcpp/libtorch/lib"],
  deps = [
    "//bark_ml/environments:gym",
    "//bark_ml/library_wrappers/lib_fqf_iqn_qrdqn/agent:agents",
  ]
)