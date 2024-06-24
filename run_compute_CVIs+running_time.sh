#!/bin/bash

# Ensure GNU Parallel is installed
if ! command -v parallel &> /dev/null
then
    echo "GNU Parallel could not be found. Please install it to run this script."
    exit 1
fi

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 /path/to/input/folder /path/to/output/folder_cvis /path/to/output/folder_time"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_FOLDER_CVIS=$2
OUTPUT_FOLDER_TIME=$3

# Create the output directories if they don't exist
mkdir -p "$OUTPUT_FOLDER_CVIS"
mkdir -p "$OUTPUT_FOLDER_TIME"

# Export the Python script path
PYTHON_SCRIPT="compute_CVIs+running_time.py"

# Function to run the Python script for a single CSV file
run_script() {
    csv_file="$1"
    output_folder_cvis="$2"
    output_folder_time="$3"
    python $PYTHON_SCRIPT "$csv_file" "$output_folder_cvis" "$output_folder_time"
}

export -f run_script
export PYTHON_SCRIPT

# Find all CSV files ending with "_predicted.csv" in all subdirectories and run the script in parallel
find "$INPUT_FOLDER" -type f -name "*_predicted.csv" | parallel run_script {} "$OUTPUT_FOLDER_CVIS" "$OUTPUT_FOLDER_TIME"
