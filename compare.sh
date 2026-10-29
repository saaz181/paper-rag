#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_runs_directory>"
    exit 1
fi

RUNS_DIR="$1"

if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Directory $RUNS_DIR does not exist."
    exit 1
fi

# Find all evaluation_results.json files in subdirectories of the runs directory
EVAL_FILES=$(find "$RUNS_DIR" -type f -name "evaluation_results.json")

if [ -z "$EVAL_FILES" ]; then
    echo "Error: No evaluation_results.json files found."
    exit 1
fi

# Prepare the output file path
OUTPUT_FILE="$RUNS_DIR/comparison.txt"

# Check if the output file already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file $OUTPUT_FILE already exists."
    read -p "Do you want to overwrite it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
    rm "$OUTPUT_FILE"
fi

# Run the compare_runs.py script with the found evaluation_results.json files
python compare_runs.py $EVAL_FILES --output "$OUTPUT_FILE"
