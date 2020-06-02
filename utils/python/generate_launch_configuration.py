from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'script_path',
    help='The name of the py file to generate a launch config for.',
    default="")


def generate(argv):
    """
    Generates a copy & paste configuration for the VS Code launch.json
    """
    path = str(FLAGS.script_path) 
    path_without_py = path if not path.endswith('.py') else path[0:-3]
    print(path_without_py)
    config = f"""
    {{
        "name": "Python: examples/tfa",
        "type": "python",
        "request": "launch",
        "program": "${{workspaceFolder}}/{path_without_py}.py\","""
    config += """
        "console": "integratedTerminal",
        "env": {
            "PYTHONPATH": \"""" \
            f"${{workspaceFolder}}/bazel-bark-ml/bark_ml/commons/:" \
            f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/:" \
            f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/bark_project/:" \
            f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/bark_project/python/:" \
            f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/bark_ml/:" \
            f"${{workspaceFolder}}/bazel-bin/{path_without_py}.runfiles/bark_ml/python/" 
    config += """"
        }
    },
    """
    print(config)

app.run(generate)
