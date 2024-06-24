#!/bin/bash

# Ensure GNU Parallel is installed
if ! command -v parallel &> /dev/null
then
    echo "GNU Parallel could not be found. Please install it to run this script."
    exit 1
fi

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/input/folder /path/to/output/folder"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Export the Python script path for parallel execution
export PYTHON_SCRIPT="visualize_dataset_shifts_gau.py"

# Run the visualization script
python $PYTHON_SCRIPT $INPUT_FOLDER $OUTPUT_FOLDER
