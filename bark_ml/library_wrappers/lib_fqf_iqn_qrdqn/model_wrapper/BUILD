cc_binary(
  name = "pytorch_script_wrapper.so",
  srcs = ["pytorch_script_wrapper.cpp"],
  deps = ["@torchcpp//:lib",
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
  deps = ["@torchcpp//:lib",],
  copts = ["-Iexternal/torchcpp/torch/include",
           #"-Lexternal/torchcpp/libtorch/lib",
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

  linkopts = ["-ltorch","-lc10"],
  linkstatic=False,
)