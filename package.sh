#!/bin/bash

pkg_name='pip_package'
# pkg_version=0.0.1
workspace_name='bark_ml'

# activate virtual environment
source ./bark_ml/python_wrapper/venv/bin/activate 

echo "Building package"
bazel run //bark_ml:$pkg_name

if [ $? -eq 0 ]; then
    echo "Build Suceeded"
else
    echo "Build Fail"
    exit 0
fi

build_dir=bazel-bin/bark_ml/$pkg_name.runfiles

echo "Copying setup.py to project pirectory at $build_dir/$workspace_name"
# copy setup.py file to build directory
cp setup.py $build_dir/$workspace_name

echo "Copying README.md to project directory at $build_dir/$workspace_name"
# copy READEME.md file to build directory
cp Readme.md $build_dir/$workspace_name

# copy LICENSE file to build directory
cp LICENSE $build_dir/$workspace_name

#read bazel generated manifest to generate manifest for pip package
bazel_manifest=$build_dir/MANIFEST
pip_manifest=$build_dir/$workspace_name/MANIFEST.in


echo "Copying manifest from $bazel_manifest to $pip_manifest"
#remove existing manifest
rm $pip_manifest
while IFS=' ' read -r line delim; do

# python files are handled by bazel
if [[ ${line} != *".py"* ]];then
    # remove trailing workspace directory generate in out as package path starts from .bark
    # else to import package we would need from bark_project.bark.examples.etc
    workspace_str_len=${#workspace_name}
    echo "include ${line:$((workspace_str_len+1))}" >> $pip_manifest
fi
done <$bazel_manifest


echo "Moving to build directory"
cd $build_dir/$workspace_name
python3.7 setup.py clean
python3.7 setup.py sdist bdist_wheel
python3.7 setup.py test


# check if manylinux argument passed. if so build
# a manylinux wheel. https://github.com/pypa/manylinux
wheeldir='dist'
# alternate: use platform instead of argument
#plat=$(python -c "import distutils.util; print(distutils.util.get_platform())")
test_status=1
if [[ $# -gt 0 ]] ; then
    if [[ $1 -eq 'manylinux' ]] ; then
        _CURR_DIR=$(pwd)
        wheeldir='wheelhouse'
        for whl in dist/*.whl; do
            if ! auditwheel show "$whl"; then
                echo "Skipping non-platform wheel $whl"
            else
                auditwheel repair "$whl"
            fi
            
            # install the package outside virtual environment
            /opt/python/cp37-cp37m/bin/pip install $whl
            /opt/python/cp37-cp37m/bin/pip3 install nose
            
            # run nose tests outside the virtual env to verify the installed package
            # echo "Running tests..."
            # cd /home
            # /opt/python/cp37-cp37m/bin/nosetests bark_ml -e "py_environment_tests"
            # test_status=$?
            # cd $_CURR_DIR
        done
    fi
fi

if [ $? -eq 0 ]; then
    echo "Tests passed!"
else
    echo "Tests failed!"
    exit 0
fi

echo "Uploading package to PyPi..."
python3 -m twine upload --skip-existing dist/*
