genrule(
  name = "generate_core",
  srcs = [
    "//bark_ml/python_wrapper:core.so",
  ],
  cmd = ("mv $(RULEDIR)/python_wrapper/core.so  $(location core.so)"),
  outs = [
    "core.so",
  ],
  visibility = ["//visibility:public"],
)

py_test(
  name = "pip_package",
  srcs = ["pip_package.py", "__init__.py"],
  data = [":generate_core"],
  deps = []
)