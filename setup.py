from setuptools import setup, find_packages, Extension
import sys, os

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules= []
if sys.platform != 'linux':
    try:
        os.mkdir('build')
    except FileExistsError:
        # directory already exists - is already created by earlier run
        pass
    open('build/temp.c','w').close()
    temp_ext = Extension('_temp', sources=['build/temp.c'])
    ext_modules.append(temp_ext)

setup(
    name = "bark-ml",
    version = "0.1.1",
    description = "Machine Learning Applied to Autonomous Driving",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers = ["Development Status :: 3 - Alpha", "Intended Audience :: Developers", "License :: OSI Approved :: MIT License", "Operating System :: POSIX", "Programming Language :: Python :: 3.7"],
    keywords = "simulator autonomous driving machine learning",
    url = "https://github.com/bark-simulator/bark-ml",
    author = "Patrick Hart, Julian Bernhard, Klemens Esterle, Tobias Kessler",
    author_email = "patrickhart.1990@gmail.com",
    license = "MIT",
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.0.3',
        'numpy>=1.18.1',
        'lxml>=4.4.2',
        'scipy>=1.4.1',
        'sphinx>=2.3.1',
        'recommonmark>=0.6.0',
        'sphinx_rtd_theme>=0.4.3',
        'pandas>=0.24.2',
        'autopep8>=1.4.4',
        'cpplint>=1.4.4',
        'pygame>=1.9.6',
        'panda3d>=1.10.5',
        'aabbtree>=2.3.1',
        'ray>=0.7.6',
        'psutil>=5.6.6',
        'notebook>=6.0.3',
        'jupyter>=1.0.0',
        'ipython>=7.13.0',
        'tf-agents>=0.5.0',
        'tensorflow>=2.2.0',
        'bark-simulator>=0.1.0'
    ],
    ext_modules=ext_modules,
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.7',
)