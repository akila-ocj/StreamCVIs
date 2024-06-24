import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import argparse

# Function to map predicted clusters to ground truth using the Hungarian algorithm
def map_clusters_to_ground_truth(labels_true, labels_pred):
    """
    Maps clustering algorithm output to ground truth labels using the Hungarian algorithm.
    :param labels_true: Ground truth labels.
    :param labels_pred: Predicted cluster labels.
    :return: Remapped predicted labels.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(labels_true, labels_pred)
    # Apply the Hungarian algorithm to the negative confusion matrix for maximum matching
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Create a new array to hold the remapped predicted labels
    remapped_labels_pred = np.zeros_like(labels_pred)
    # For each original cluster index, find the new label (according to the Hungarian algorithm)
    # and assign it in the remapped labels array
    for original_cluster, new_label in zip(col_ind, row_ind):
        remapped_labels_pred[labels_pred == original_cluster] = new_label

    return remapped_labels_pred

def get_dataset_names(directory):
    """Return a list of dataset names (folder names) in the given directory."""
    return sorted([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

def read_csv_files(dir_path, dataset_name):
    """Read the CSV files for the given dataset from the specified directory."""
    csv_files = [file for file in os.listdir(dir_path) if file.endswith('.csv') and dataset_name in file]
    csv_files.sort()  # Ensure files are sorted to pick the last one for each condition
    return csv_files

def visualize_labels(true_data, birch_data, streamkmeans_data, dbstream_data, output_path):
    """Visualize the true and predicted labels for the given datasets."""
    plt.figure(figsize=(12, 10))

    # Define colors for each label
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Plot True Labels (common for all)
    plt.subplot(2, 2, 1)
    for label in sorted(true_data['TrueLabel'].unique()):
        subset = true_data[true_data['TrueLabel'] == label]
        plt.scatter(subset['Feature_0'], subset['Feature_1'], color=colors[label % len(colors)], label=f'True Label {label}')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('True Labels')
    plt.grid(True)

    # Plot Birch Predicted Labels
    plt.subplot(2, 2, 2)
    for label in sorted(birch_data['MappedPredictedLabel'].unique()):
        subset = birch_data[birch_data['MappedPredictedLabel'] == label]
        plt.scatter(subset['Feature_0'], subset['Feature_1'], color=colors[label % len(colors)])
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('BIRCH Predicted Labels')
    plt.grid(True)

    # Plot StreamKMeans Predicted Labels
    plt.subplot(2, 2, 3)
    for label in sorted(streamkmeans_data['MappedPredictedLabel'].unique()):
        subset = streamkmeans_data[streamkmeans_data['MappedPredictedLabel'] == label]
        plt.scatter(subset['Feature_0'], subset['Feature_1'], color=colors[label % len(colors)])
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('StreamKMeans Predicted Labels')
    plt.grid(True)

    # Plot DBStream Predicted Labels
    plt.subplot(2, 2, 4)
    for label in sorted(dbstream_data['MappedPredictedLabel'].unique()):
        subset = dbstream_data[dbstream_data['MappedPredictedLabel'] == label]
        plt.scatter(subset['Feature_0'], subset['Feature_1'], color=colors[label % len(colors)])
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title('DBStream Predicted Labels')
    plt.grid(True)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(birch_dir, dbstream_dir, stream_kmeans_dir, output_dir):
    # Get dataset names from each directory
    birch_datasets = get_dataset_names(birch_dir)
    dbstream_datasets = get_dataset_names(dbstream_dir)
    stream_kmeans_datasets = get_dataset_names(stream_kmeans_dir)

    # Ensure the datasets are the same across all directories
    if not (birch_datasets == dbstream_datasets == stream_kmeans_datasets):
        raise ValueError("Datasets do not match across directories!")

    datasets = birch_datasets  # Since all are the same, we can use any

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each dataset
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")

        # Read the CSV files for each dataset condition
        birch_files = read_csv_files(os.path.join(birch_dir, dataset_name), dataset_name)
        dbstream_files = read_csv_files(os.path.join(dbstream_dir, dataset_name), dataset_name)
        streamkmeans_files = read_csv_files(os.path.join(stream_kmeans_dir, dataset_name), dataset_name)

        if not (birch_files and dbstream_files and streamkmeans_files):
            print(f"Skipping {dataset_name} due to missing files.")
            continue

        # Filter dataset conditions, including the no-shift condition
        conditions = ['no_shift', 'mild_gaussian-noise', 'mild_knock-out', 'moderate_gaussian-noise', 'moderate_knock-out', 'severe_gaussian-noise', 'severe_knock-out']

        for condition in conditions:
            if condition == 'no_shift':
                condition_birch_files = [file for file in birch_files if 'transformed_' + dataset_name + '.csv_predicted.csv' in file]
                condition_dbstream_files = [file for file in dbstream_files if 'transformed_' + dataset_name + '.csv_predicted.csv' in file]
                condition_streamkmeans_files = [file for file in streamkmeans_files if 'transformed_' + dataset_name + '.csv_predicted.csv' in file]
            else:
                condition_birch_files = [file for file in birch_files if condition in file]
                condition_dbstream_files = [file for file in dbstream_files if condition in file]
                condition_streamkmeans_files = [file for file in streamkmeans_files if condition in file]

            if not (condition_birch_files and condition_dbstream_files and condition_streamkmeans_files):
                print(f"Skipping condition {condition} for dataset {dataset_name} due to missing files.")
                continue

            # Read the last file (assumed to be the most recent step) for each algorithm
            birch_df = pd.read_csv(os.path.join(birch_dir, dataset_name, condition_birch_files[-1]))
            dbstream_df = pd.read_csv(os.path.join(dbstream_dir, dataset_name, condition_dbstream_files[-1]))
            streamkmeans_df = pd.read_csv(os.path.join(stream_kmeans_dir, dataset_name, condition_streamkmeans_files[-1]))

            # Get the true labels
            true_labels = birch_df[['Feature_0', 'Feature_1', 'TrueLabel']].copy()

            # Map predicted labels to true labels for consistent coloring
            birch_df['MappedPredictedLabel'] = map_clusters_to_ground_truth(birch_df['TrueLabel'], birch_df['PredictedLabel'])
            dbstream_df['MappedPredictedLabel'] = map_clusters_to_ground_truth(dbstream_df['TrueLabel'], dbstream_df['PredictedLabel'])
            streamkmeans_df['MappedPredictedLabel'] = map_clusters_to_ground_truth(streamkmeans_df['TrueLabel'], streamkmeans_df['PredictedLabel'])


            # Visualize the true and predicted labels
            output_path = os.path.join(output_dir, f'visualization_{dataset_name}_{condition}.png')
            visualize_labels(true_labels, birch_df, streamkmeans_df, dbstream_df, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize clustering results for datasets.')
    parser.add_argument('birch_dir', type=str, help='Directory containing BIRCH predicted label CSV files')
    parser.add_argument('dbstream_dir', type=str, help='Directory containing DBSTREAM predicted label CSV files')
    parser.add_argument('stream_kmeans_dir', type=str, help='Directory containing STREAM_KMEANS predicted label CSV files')
    parser.add_argument('output_dir', type=str, help='Directory to save the output plots')

    args = parser.parse_args()

    main(args.birch_dir, args.dbstream_dir, args.stream_kmeans_dir, args.output_dir)
