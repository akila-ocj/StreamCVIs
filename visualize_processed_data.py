import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import numpy as np

def generate_color_gradient(base_colors, num_colors):
    if num_colors <= len(base_colors):
        return base_colors[:num_colors]
    else:
        base_cmap = mcolors.LinearSegmentedColormap.from_list("base_cmap", base_colors, N=len(base_colors))
        return [base_cmap(i / (num_colors - 1)) for i in range(num_colors)]

def visualize_file(input_file, output_folder):
    # Read CSV file into a DataFrame
    df = pd.read_csv(input_file, header=None)

    # Extract features and cluster labels
    features = df.iloc[:, :-2]
    cluster_labels = df.iloc[:, -2]
    dataset_labels = df.iloc[:, -1]

    # Define colors for each label
    base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    unique_labels = cluster_labels.unique()
    colors = generate_color_gradient(base_colors, len(unique_labels))

    # Create a color map based on cluster labels
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Prepare the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    fig.suptitle(f'Visualization for {os.path.basename(input_file)}')

    # Define the dataset types
    dataset_types = ['train', 'validate', 'test']

    # Plot each dataset type in a separate subplot
    for i, dataset_type in enumerate(dataset_types):
        subset = df[df.iloc[:, -1] == dataset_type]
        scatter = axes[i].scatter(subset.iloc[:, 0], subset.iloc[:, 1],
                                  c=subset.iloc[:, -2].map(color_map), alpha=0.6)
        axes[i].set_title(dataset_type.capitalize())
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')

    # Add a color bar
    norm = mcolors.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))
    sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(colors), norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)

    # Prepare output file path
    output_file_path = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(input_file))[0]}_visualization.png')

    # Save the plot to the output folder
    plt.savefig(output_file_path)
    plt.close()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Visualize processed data with clusters for a single CSV file.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where the visualizations will be saved.')

    args = parser.parse_args()

    # Visualize the file
    visualize_file(args.input_file, args.output_folder)
