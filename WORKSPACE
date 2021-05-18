workspace(name = "bark_ml")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

load("//utils:dependencies.bzl", "bark_ml_dependencies")
bark_ml_dependencies()

load("@bark_project//tools:deps.bzl", "bark_dependencies")
bark_dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

git_repository(
  name = "pybind11_bazel",
  commit="26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
  remote = "https://github.com/pybind/pybind11_bazel"
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

# -------- Benchmark Database -----------------------
git_repository(
  name = "benchmark_database",
  commit="ff6e433ecb7878ebe59996f3994ff67483a7c297",
  remote = "https://github.com/bark-simulator/benchmark-database"
)

load("@benchmark_database//util:deps.bzl", "benchmark_database_dependencies")
load("@benchmark_database//load:load.bzl", "benchmark_database_release")
benchmark_database_dependencies()
benchmark_database_release()
# --------------------------------------------------
