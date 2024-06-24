#!/bin/bash

# Ensure GNU Parallel is installed
if ! command -v parallel &> /dev/null
then
    echo "GNU Parallel could not be found. Please install it to run this script."
    exit 1
fi

# Check if the correct number of arguments are provided
# usage: ./run_apply_dataset_shift_gau.sh processed_data/ processed_data+dataset_shift 0.5
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 /path/to/input/folder /path/to/output/folder [fraction]"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2
FRACTION=${3:-0.1}  # Default to 0.1 if not provided

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Export the Python script path for parallel execution
export PYTHON_SCRIPT="apply_dataset_shift_gau.py"

# Find all CSV files in the input folder and process them in parallel
find "$INPUT_FOLDER" -name '*.csv' | parallel python $PYTHON_SCRIPT {} "$OUTPUT_FOLDER" --frac "$FRACTION"
