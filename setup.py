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
    version = "0.2.2",
    description = "Machine Learning Applied to Autonomous Driving",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers = ["Development Status :: 4 - Beta",
                   "Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python :: 3.7"],
    keywords = "simulator autonomous driving machine learning",
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
        'bark-simulator>=0.1.0',
        'tf2_gnn>=2.4.0',
        'spektral>0.6.0'
    ],
    ext_modules=ext_modules,
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.7',
)
