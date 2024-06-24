import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
from sklearn.metrics import confusion_matrix

def get_datasets(directory):
    """Return a list of datasets (folder names) in the given directory."""
    return sorted([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

def truncate_name(name, length=14):
    """Truncate dataset name to a maximum length."""
    return name if len(name) <= length else name[:length]

def calculate_purity(true_labels, predicted_labels):
    """Calculate the purity score."""
    contingency_matrix = confusion_matrix(true_labels, predicted_labels)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def read_predicted_labels(dir_path, dataset_name, condition):
    """Read the CSV files containing the predicted labels for the given dataset and condition from the specified directory."""
    if condition == 'no_shift':
        csv_file = f"transformed_{dataset_name}.csv_predicted.csv"
    else:
        csv_file = f"transformed_{dataset_name}_{condition}.csv_predicted.csv"
    return csv_file if os.path.exists(os.path.join(dir_path, dataset_name, csv_file)) else None

def main(birch_dir, dbstream_dir, stream_kmeans_dir, output_dir):
    # Define the paths to the directories containing the datasets for each algorithm
    dirs = {
        "BIRCH": birch_dir,
        "DBSTREAM": dbstream_dir,
        "STREAM_KMEANS": stream_kmeans_dir
    }

    # Get datasets for each algorithm
    datasets_birch = get_datasets(dirs["BIRCH"])
    datasets_dbstream = get_datasets(dirs["DBSTREAM"])
    datasets_stream_kmeans = get_datasets(dirs["STREAM_KMEANS"])

    # Ensure the datasets are the same across all directories
    if not (datasets_birch == datasets_dbstream == datasets_stream_kmeans):
        raise ValueError("Datasets do not match across directories!")

    # Use one of the datasets lists since they are all the same
    datasets = datasets_birch

    # Truncate dataset names
    truncated_datasets = [truncate_name(dataset) for dataset in datasets]

    # Define dataset conditions
    conditions = ['no_shift', 'mild_gaussian-noise', 'mild_knock-out', 'moderate_gaussian-noise', 'moderate_knock-out', 'severe_gaussian-noise', 'severe_knock-out']

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each condition
    for condition in conditions:
        purity_scores = {dataset: {} for dataset in datasets}

        # Read purity scores from predicted label files
        for algo, dir_path in dirs.items():
            for dataset in datasets:
                csv_file = read_predicted_labels(dir_path, dataset, condition)
                if csv_file:
                    # Read the predicted label CSV file and calculate purity
                    df = pd.read_csv(os.path.join(dir_path, dataset, csv_file))
                    purity = calculate_purity(df['TrueLabel'], df['PredictedLabel'])
                    purity_scores[dataset][algo] = purity
                else:
                    purity_scores[dataset][algo] = 0  # Handle missing files by setting purity to 0

        # Set up the plot
        fig, ax = plt.subplots(figsize=(16, 9))

        # Define positions and width for bars
        bar_width = 0.2
        indices = np.arange(len(datasets))

        # Define color palette
        colors = sns.color_palette("Set2", n_colors=3)

        # Plot each algorithm's purity scores
        for i, (algo, color) in enumerate(zip(dirs.keys(), colors)):
            purities = [purity_scores[dataset][algo] for dataset in datasets]
            ax.bar(indices + i * bar_width, purities, bar_width, label=algo, color=color)

            # Calculate and plot the mean purity score as a dotted line
            mean_purity = np.mean(purities)
            ax.axhline(y=mean_purity, color=color, linestyle='--', linewidth=2)
            ax.text(len(datasets) + 0.9, mean_purity, f'{algo}: {mean_purity:.2f}', color=color, va='center',
                    fontsize=14, fontweight='bold')

        # Set the position of the x ticks and labels
        ax.set_xticks(indices + bar_width)
        ax.set_xticklabels(truncated_datasets, rotation=45, ha='right', fontsize=12, fontweight='bold')

        # Set labels and title
        ax.set_xlabel('Datasets', fontsize=14, fontweight='bold')
        ax.set_ylabel('Purity Score', fontsize=14, fontweight='bold')
        ax.set_title(f'Purity Scores for Different Clustering Algorithms - {condition}', fontsize=18, fontweight='bold')

        # Improve layout
        plt.ylim(0, 1)  # Assuming purity scores are between 0 and 1
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'purity_scores_histogram_{condition}.png')
        plt.savefig(output_path, dpi=300)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot purity scores for clustering algorithms.')
    parser.add_argument('birch_dir', type=str, help='Directory containing datasets for BIRCH algorithm')
    parser.add_argument('dbstream_dir', type=str, help='Directory containing datasets for DBSTREAM algorithm')
    parser.add_argument('stream_kmeans_dir', type=str, help='Directory containing datasets for STREAM_KMEANS algorithm')
    parser.add_argument('output_dir', type=str, help='Directory to save the output plots')

    args = parser.parse_args()

    main(args.birch_dir, args.dbstream_dir, args.stream_kmeans_dir, args.output_dir)
