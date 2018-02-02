#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

cd $ROOT_DIR

export ROOT_DIR=$ROOT_DIR

echo "PYTHON PATH IS: $PYTHONPATH"