#!/bin/bash

# Ensure GNU Parallel is installed
if ! command -v parallel &> /dev/null
then
    echo "GNU Parallel could not be found. Please install it to run this script."
    exit 1
fi

# Check if the correct number of arguments are provided
# usage: ./run_apply_dataset_shift_knock.sh processed_data/ processed_data+dataset_shift half
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 /path/to/input/folder /path/to/output/folder one|half"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2
CLASS_FRACTION=$3

# Validate the third argument
if [[ "$CLASS_FRACTION" != "one" && "$CLASS_FRACTION" != "half" ]]; then
    echo "The third argument must be either 'one' or 'half'"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Export the Python script path for parallel execution
export PYTHON_SCRIPT="apply_dataset_shift_knock.py"

# Find all CSV files in the input folder and process them in parallel
find "$INPUT_FOLDER" -name '*.csv' | parallel python $PYTHON_SCRIPT {} "$OUTPUT_FOLDER" "$CLASS_FRACTION"
