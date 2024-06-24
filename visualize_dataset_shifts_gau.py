import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse


def generate_color_gradient(base_colors, num_colors):
    if num_colors <= len(base_colors):
        return base_colors[:num_colors]
    else:
        base_cmap = mcolors.LinearSegmentedColormap.from_list("base_cmap", base_colors, N=len(base_colors))
        return [base_cmap(i / (num_colors - 1)) for i in range(num_colors)]


def visualize_file(dataset_name, input_folder, output_folder):
    # Define noise levels and corresponding file suffixes
    noise_levels = ['mild', 'moderate', 'severe']
    file_suffixes = ['mild_gaussian-noise.csv', 'moderate_gaussian-noise.csv', 'severe_gaussian-noise.csv']

    # Define base colors for the clusters
    base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan', 'magenta', 'yellow', 'black', 'navy', 'lime', 'teal'
    ]



    # Prepare the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    fig.suptitle(f'Gaussian noise: {dataset_name}')

    for i, (level, suffix) in enumerate(zip(noise_levels, file_suffixes)):
        input_file = os.path.join(input_folder, f'{dataset_name}_{suffix}')

        if not os.path.exists(input_file):
            print(f'File {input_file} does not exist. Skipping.')
            continue

        # Read CSV file into a DataFrame
        df = pd.read_csv(input_file, header=None)

        # Filter to include only the test set
        df = df[df.iloc[:, -1] == 'test']

        # Extract features and cluster labels
        features = df.iloc[:, :-2]
        cluster_labels = df.iloc[:, -2]
        dataset_labels = df.iloc[:, -1]

        # Generate colors for clusters
        unique_labels = cluster_labels.unique()
        colors = generate_color_gradient(base_colors, len(unique_labels))

        # Create a color map based on cluster labels
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

        # Plot the data
        scatter = axes[i].scatter(df.iloc[:, 0], df.iloc[:, 1], c=cluster_labels.map(color_map), alpha=0.6)
        axes[i].set_title(f'{level.capitalize()} Noise')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')

    # Add a color bar
    norm = mcolors.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))
    sm = plt.cm.ScalarMappable(cmap=mcolors.ListedColormap(colors), norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)

    # Prepare output file path
    output_file_path = os.path.join(output_folder, f'{dataset_name}_visualization_gau.png')

    # Save the plot to the output folder
    plt.savefig(output_file_path)
    plt.close()


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Visualize dataset shifts with Gaussian noise.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing CSV files.')
    parser.add_argument('output_folder', type=str,
                        help='Path to the output folder where the visualizations will be saved.')

    args = parser.parse_args()

    # Process each dataset in the input folder
    datasets = set('_'.join(f.split('_')[:2]) for f in os.listdir(args.input_folder) if f.endswith('.csv'))

    for dataset_name in datasets:
        visualize_file(dataset_name, args.input_folder, args.output_folder)
