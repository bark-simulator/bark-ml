py_test(
  name = "continuous_env",
  srcs = ["continuous_env.py"],
  data = ["@bark_project//python:bark.so",
          "//python:bark_ml_library.so"],
  imports = ['../python/'],
  deps = [
    "//bark_ml/environments:gym",
  ]
)

py_test(
  name = "discrete_env",
  srcs = ["discrete_env.py"],
  data = ["@bark_project//python:bark.so",
          "//python:bark_ml_library.so"],
  imports = ['../python/'],
  deps = [
    "//bark_ml/environments:gym",
  ]
)