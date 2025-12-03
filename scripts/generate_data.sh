#!/bin/bash

# Script to run the data generation process

# Ensure we are in the project root
# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Running create_examples.py from project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Run the python script
python scripts/create_examples.py

echo "Done."
