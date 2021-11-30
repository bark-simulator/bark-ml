from setuptools import setup, find_packages, Extension
import os

from setuptools import setup, find_packages, Extension
import os,sys
import os
import shlex
import shutil
import setuptools.command.build_ext
import setuptools.command.build_py
import setuptools.command.install
import setuptools.command.sdist
import setuptools.dist
from setuptools import dist
from setuptools.command.install import install
import sysconfig
import tempfile
import pkg_resources
from distutils.command.build import build

with open("Readme.md", "r") as fh:
    long_description = fh.read()

def _configure_macos_deployment_target():
  # TensorStore requires MACOSX_DEPLOYMENT_TARGET >= 10.14 in
  # order to support sized/aligned operator new/delete.
  min_macos_target = '10.14'
  key = 'MACOSX_DEPLOYMENT_TARGET'
  python_macos_target = str(sysconfig.get_config_var(key))
  macos_target = python_macos_target
  if (macos_target and (pkg_resources.parse_version(macos_target) <
                        pkg_resources.parse_version(min_macos_target))):
    macos_target = min_macos_target

  # macos_target_override = os.getenv(key)
  # if macos_target_override:
  #   if (pkg_resources.parse_version(macos_target_override) <
  #       pkg_resources.parse_version(macos_target)):
  #     print('%s=%s is set in environment but >= %s is required by this package '
  #           'and >= %s is required by the current Python build' %
  #           (key, macos_target_override, min_macos_target, python_macos_target))
  #     sys.exit(1)
  #   else:
  #     macos_target = macos_target_override

  # Set MACOSX_DEPLOYMENT_TARGET in the environment, because the `wheel` package
  # checks there.  Note that Bazel receives the version via a command-line
  # option instead.
  os.environ[key] = macos_target
  return macos_target

if 'darwin' in sys.platform:
  _macos_deployment_target = _configure_macos_deployment_target()

class CustomBuild(build):
  def run(self):
    self.build_lib = '_build'

try:
  from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
  class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
      _bdist_wheel.finalize_options(self)
      self.root_is_pure = False
except ImportError:
  bdist_wheel = None

class BinaryDistribution(dist.Distribution):
  def is_pure(self):
    return False
  def has_ext_modules(self):
    return True

class InstallPlatlib(install):
  def finalize_options(self):
    install.finalize_options(self)
    if self.distribution.has_ext_modules():
      self.install_lib = self.install_platlib

class BuildExtCommand(setuptools.command.build_ext.build_ext):
  """Overrides default build_ext command to invoke bazel."""

  def run(self):
    if not self.dry_run:
      prebuilt_path = os.getenv('BARK_ML_PREBUILT_DIR')
      if not prebuilt_path:
        # Ensure python_configure.bzl finds the correct Python verison.
        os.environ['PYTHON_BIN_PATH'] = sys.executable
        bazelisk = os.getenv('BARK_ML_BAZELISK', 'bazelisk.py')
        # Controlled via `setup.py build_ext --debug` flag.
        default_compilation_mode = 'dbg' if self.debug else 'opt'
        compilation_mode = os.getenv('BARK_ML_BAZEL_COMPILATION_MODE',
                                     default_compilation_mode)
        build_command = [sys.executable, '-u', bazelisk] + [
            'build',
            '-c',
            compilation_mode,
            '//bark_ml:pip_package',
            '--verbose_failures'
        ]
        if 'darwin' in sys.platform:
          # Note: Bazel does not use the MACOSX_DEPLOYMENT_TARGET environment
          # variable.
          build_command += ['--macos_minimum_os=%s' % _macos_deployment_target]
          build_command += ['--define=build_platform=macos']
        if sys.platform == 'win32':
          # Disable newer exception handling from Visual Studio 2019, since it
          # requires a newer C++ runtime than shipped with Python.
          #
          # https://cibuildwheel.readthedocs.io/en/stable/faq/#importerror-dll-load-failed-the-specific-module-could-not-be-found-error-on-windows
          build_command += ['--copt=/d2FH4-']

        self.spawn(build_command)
        suffix = '.pyd' if os.name == 'nt' else '.so'
        built_ext_path = os.path.join(
          'bazel-bin/bark_ml/pip_package.runfiles/bark_ml/bark_ml/core' + suffix)

      # os.makedirs(os.path.dirname(ext_full_path), exist_ok=True)
      copy_to = os.path.dirname(os.path.abspath(__file__)) + "/bark_ml/core.so"
      copy_to_manifest = os.path.dirname(os.path.abspath(__file__)) + "/MANIFEST.in"
      print('Copying extension %s -> %s' % (
          built_ext_path,
          copy_to
      ))
      shutil.copyfile(built_ext_path, copy_to)

setup(
    name = "bark-ml",
    version = "0.4.26",
    description = "Gym Environments and Agents for Autonomous Driving",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers = ["Development Status :: 4 - Beta",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8"],
    keywords = "gym environments, reinforcement learning, autonomous driving, machine learning",
    url = "https://github.com/bark-simulator/bark-ml",
    author = "Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler",
    author_email = "patrickhart.1990@gmail.com",
    license = "MIT",
    packages=find_packages(),
    install_requires=[
        'pygame>=1.9.6',
        'gym>=0.17.2',
        'tensorflow>=2.2.0',
        'tensorboard>=2.2.2',
        'tf-agents>=0.5.0',
        'tensorflow-probability>=0.10.0',
        'bark-simulator>=1.4.8',
        'graph-nets>=1.1.0'
    ],
    cmdclass={
      'bdist_wheel': bdist_wheel,
      'build_ext': BuildExtCommand,
      'install': InstallPlatlib,
      'build': CustomBuild
    },
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    python_requires='>=3.7',
)
