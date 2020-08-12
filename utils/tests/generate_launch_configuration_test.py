from generate_launch_configuration import generate
import unittest


class TestCorrectGenerated(unittest.TestCase):
  """
  Tests for the generate launch json script
  """

  def test_example_tfa_correct_generated(self):
    """
    Test: Generate the config for example/tfa.py
    """

    # Dont reformat this multi line string!!!
    # All strange formatting is intended.
    expected = """
    {
        "name": "Python: examples/tfa",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/examples/tfa.py",
        "console": "integratedTerminal",
        "env": {
            "PYTHONPATH": "${workspaceFolder}/bazel-bin/examples/tfa.runfiles/:${workspaceFolder}/bazel-bin/examples/tfa.runfiles/bark_project/:${workspaceFolder}/bazel-bin/examples/tfa.runfiles/bark_project/python/:${workspaceFolder}/bazel-bin/examples/tfa.runfiles/bark_ml/:${workspaceFolder}/bazel-bin/examples/tfa.runfiles/bark_ml/python/:${workspaceFolder}/bazel-bin/examples/tfa.runfiles/com_github_interaction_dataset_interaction_dataset/python/:${workspaceFolder}/bazel-bin/examples/tfa.runfiles/com_github_keiohta_tf2rl"
        }
    },"""
    actual = generate("examples/tfa")
    print(actual)
    self.assertEqual(actual.strip(), expected.strip())

  def test_no_path_given(self):
    """
    Test: Generate generic version if no path given.
    """

    # Dont reformat this multi line string!!!
    # All strange formatting is intended.
    expected = """
    {
        "name": "Python: <ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>",
        "type": "python",
        "request": "launch",
        "program": "${workspaceFolder}/<ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>.py",
        "console": "integratedTerminal",
        "env": {
            "PYTHONPATH": "${workspaceFolder}/bazel-bin/<ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>.runfiles/:${workspaceFolder}/bazel-bin/<ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>.runfiles/bark_project/:${workspaceFolder}/bazel-bin/<ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>.runfiles/bark_project/python/:${workspaceFolder}/bazel-bin/<ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>.runfiles/bark_ml/:${workspaceFolder}/bazel-bin/<ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>.runfiles/bark_ml/python/:${workspaceFolder}/bazel-bin/<ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>.runfiles/com_github_interaction_dataset_interaction_dataset/python/:${workspaceFolder}/bazel-bin/<ADD-PYTHON-EXECUTABLE-PATH-WITHOUT-.PY>.runfiles/com_github_keiohta_tf2rl"
        }
    },"""
    self.assertEqual(generate('').strip(), expected.strip())
    self.assertEqual(generate().strip(), expected.strip())

  def test_same_with_and_without_py_ending(self):
    """
    Test: We can omit the .py file ending when specifying a script path.
    """
    self.assertEqual(generate("examples/tfa"), generate("examples/tfa.py"))


if __name__ == '__main__':
  unittest.main()
