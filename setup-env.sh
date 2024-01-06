#!/bin/bash


[ -z "$1" ] && echo "Please provide a name for the virtual environment" && exit 1

# create virtual environment
VENV_NAME="$1"

python -m venv "$VENV_NAME"
source "$VENV_NAME"/bin/activate

# install source code dependencies
pip install pysindy numpy pyGPGO matplotlib scikit-learn scipy pathlib pandas cvxpy gurobipy

# activate virtual environment and run
echo "Virtual environment created. Activate with (see below):"
echo "source $VENV_NAME/bin/activate"
