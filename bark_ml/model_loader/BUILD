cc_library(
    name="model_loader",
    srcs=["model_loader.cpp"],
    hdrs=["model_loader.hpp"],
    copts = ["-Iexternal/libtensorflow-RL-MCTS"],
    deps= ["@libtensorflow-RL-MCTS//libtensorflow:tensorflow_include"],
    visibility = ["//visibility:public"],
)