import os
import shutil
import argparse
from collections import defaultdict

def find_and_group_csv_files(input_folder, output_folder):
    # Create a dictionary to hold lists of files for each dataset
    datasets = defaultdict(list)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            # Split the filename to find the dataset name
            parts = filename.split('_')
            if len(parts) > 1:
                dataset_name = parts[1].split('.')[0]
                datasets[dataset_name].append(filename)

    # Create directories and move files to the respective folders
    for dataset_name, files in datasets.items():
        # Create the output directory for the dataset if it does not exist
        dataset_folder = os.path.join(output_folder, dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)

        # Move each file to the dataset folder
        for file in files:
            src_path = os.path.join(input_folder, file)
            dest_path = os.path.join(dataset_folder, file)
            shutil.move(src_path, dest_path)
            print(f'Moved {file} to {dataset_folder}')

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Find and group CSV files by dataset name.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing CSV files.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to save grouped files.')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    find_and_group_csv_files(args.input_folder, args.output_folder)
