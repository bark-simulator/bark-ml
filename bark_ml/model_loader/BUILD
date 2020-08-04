

cc_library(
    name="ModelLoader",
    srcs=["ModelLoader.cpp"],
    hdrs=["ModelLoader.hpp"],
    copts = ["-Iexternal/libtensorflow-RL-MCTS"],
    deps= ["@libtensorflow-RL-MCTS//libtensorflow:tensorflow_include"],
    visibility = ["//visibility:public"],
)