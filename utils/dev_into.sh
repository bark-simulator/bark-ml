#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENTDIR=$(dirname "$DIR")
source $PARENTDIR/bark_ml/python_wrapper/venv/bin/activate
export LD_LIBRARY_PATH=$PARENTDIR/bark_ml/python_wrapper/venv/lib/python3.7/site-packages/torch/lib/ 
