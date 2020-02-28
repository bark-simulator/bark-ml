load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def _maybe(repo_rule, name, **kwargs):
    if name not in native.existing_rules():
        repo_rule(name = name, **kwargs)

def load_bark():
  # _maybe(
  #   native.local_repository,
  #   name = "bark_project",
  #   path="/home/hart/Dokumente/2020/bark",
  # )
  _maybe(
    git_repository,
    name = "bark_project",
    commit="e13042163625fdb7c5332da195f8d88d9bd70845",
    remote = "https://github.com/bark-simulator/bark",
  )
