#!/bin/bash

# Ensure GNU Parallel is installed
if ! command -v parallel &> /dev/null
then
    echo "GNU Parallel could not be found. Please install it to run this script."
    exit 1
fi

# Check if the correct number of arguments are provided
# usage: ./run_clustering_stream_kmeans.sh grouped_processed_data+dataset_shift/ predicted_stream_kmeans
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/input/folder /path/to/output/folder"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Export the Python script path
PYTHON_SCRIPT="clustering_stream_kmeans.py"

# Function to run the Python script for a single directory
run_script() {
    dir="$1"
    dir_name=$(basename "$dir")
    output_dir="$OUTPUT_FOLDER/$dir_name"
    mkdir -p "$output_dir"
    python $PYTHON_SCRIPT "$dir" "$output_dir"
}

export -f run_script
export PYTHON_SCRIPT
export OUTPUT_FOLDER

# Find all subdirectories and run the script in parallel
find "$INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type d | parallel run_script {}
