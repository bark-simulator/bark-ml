from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'script_path',
    help='The name of the py file to generate a launch config for.',
    default="")


def generate(script_path=None):
  """
  Generates a copy & paste configuration for the VS Code launch.json
  """
  if not script_path:
    script_path = ""

  path = str(script_path)
  path_without_py = path if not path.endswith('.py') else path[0:-3]

  if not path_without_py:
    path_without_py = '<ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>'

  config = f"""
    {{
        "name": "Python: {path_without_py}",
        "type": "python",
        "request": "launch",
        "program": "${{workspaceFolder}}/{path_without_py}.py\","""
  config += """
        "console": "integratedTerminal",
        "env": {
            "PYTHONPATH": \"""" \
          f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/:" \
          f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/bark_project/:" \
          f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/bark_project/python/:" \
          f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/bark_ml/:" \
          f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/bark_ml/python/:" \
          f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/com_github_interaction_dataset_interaction_dataset/python/:"\
          f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/com_github_keiohta_tf2rl"
  config += """"
        }
    },"""
  return config


def main(argv):
  """ main """
  print(generate(FLAGS.script_path))


if __name__ == '__main__':
  app.run(main)
