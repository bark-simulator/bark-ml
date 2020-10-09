filegroup(
    name = "eval_data",
    srcs = glob(["data/*.pckl"]),
    visibility = ["//visibility:public"],
)

py_test(
  name = "plot_stats",
  srcs = ["plot_stats.py"],
  data = [":eval_data"],
  imports = [""],
  deps = [
    "//bark_ml/commons:tracer",
  ]
)