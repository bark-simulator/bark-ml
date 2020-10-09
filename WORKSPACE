workspace(name = "bark_ml")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

load("//utils:dependencies.bzl", "bark_ml_dependencies")
bark_ml_dependencies()

load("@bark_project//tools:deps.bzl", "bark_dependencies")
bark_dependencies()

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")
boost_deps()

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