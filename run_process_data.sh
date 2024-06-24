#!/bin/bash

# Ensure GNU Parallel is installed
if ! command -v parallel &> /dev/null
then
    echo "GNU Parallel could not be found. Please install it to run this script."
    exit 1
fi

# Check if the correct number of arguments are provided
# usage: ./run_process_data.sh data/ processed_data
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/input/folder /path/to/output/folder"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Export the Python script path for parallel execution
export PYTHON_SCRIPT="process_data.py"

# Find all CSV files in the input folder and process them in parallel
find "$INPUT_FOLDER" -name '*.csv' | parallel python $PYTHON_SCRIPT {} "$OUTPUT_FOLDER"
