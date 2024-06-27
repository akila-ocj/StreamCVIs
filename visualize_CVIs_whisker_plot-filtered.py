import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Define internal and external CVI metrics and their acronyms
internal_cvi_metrics = [
    'davies_bouldin_score', 'iDB', 'iWB', 'iXB', 'irCIP', 'silhouette_score'
]

external_cvi_metrics = [
    'adjusted_rand_index', 'jaccard_index', 'fowlkes_mallows_index', 'homogeneity_score',
    'normalized_mutual_info_score', 'v_measure_score'
]

# Mapping of CVI metric names to their acronyms
cvi_acronyms = {
    'davies_bouldin_score': 'DB',
    'silhouette_score': 'Sil',
    'adjusted_rand_index': 'ARI',
    'jaccard_index': 'Jaccard',
    'fowlkes_mallows_index': 'FMI',
    'homogeneity_score': 'HomoG',
    'normalized_mutual_info_score': 'NMI',
    'v_measure_score': 'V-measure',
    'iDB': 'iDB',
    'iWB': 'iWB',
    'iXB': 'iXB',
    'irCIP': 'irCIP'
}

def process_csv(file_path, output_dir, cvi_type):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Replace '--' with np.nan to handle as nulls
    df.replace('--', np.nan, inplace=True)

    # Drop the first column (Dataset name)
    df = df.iloc[:, 1:]

    # Select the CVI metrics based on the type
    if cvi_type == 'internal':
        cvi_metrics = internal_cvi_metrics
    elif cvi_type == 'external':
        cvi_metrics = external_cvi_metrics
    else:
        raise ValueError(f"Unknown CVI type: {cvi_type}")

    # Prepare data for plotting
    plot_data = []

    for cvi in cvi_metrics:
        for algo in ["BIRCH", "DBSTREAM", "STREAM_KMEANS"]:
            col_name = f"{cvi}_{algo}"
            if col_name in df.columns:
                # Drop rows with np.nan values in the specific column
                df_clean = df.dropna(subset=[col_name])
                for value in df_clean[col_name]:
                    plot_data.append({
                        "CVI": cvi_acronyms.get(cvi, cvi),  # Use acronym
                        "Algorithm": algo,
                        "Value": float(value)  # Ensure value is treated as a float
                    })

    plot_df = pd.DataFrame(plot_data)

    # Ensure that 'CVI' and 'Algorithm' are treated as categorical data
    plot_df['CVI'] = pd.Categorical(plot_df['CVI'], categories=[cvi_acronyms.get(cvi, cvi) for cvi in cvi_metrics], ordered=True)
    plot_df['Algorithm'] = plot_df['Algorithm'].astype('category')

    # Set up the plot with dimensions suitable for the red box
    plt.figure(figsize=(4, 3))  # Adjusted dimensions to fit the red box area
    sns.boxplot(x="CVI", y="Value", hue="Algorithm", data=plot_df, palette="Set2")
    plt.xticks(rotation=0, fontsize=6)  # Reduced font size for x-axis values
    plt.yticks(fontsize=6)  # Reduced font size for y-axis values
    plt.xlabel('CVI', fontsize=0)  # Reduced font size for x-axis label
    plt.ylabel('Correlation with Purity', fontsize=5)  # Reduced font size for y-axis label
    plt.legend(prop={'size': 6})  # Reduced legend font size

    # Add horizontal dotted lines at every 0.25 correlation value
    for y in np.arange(-1, 1.25, 0.25):
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

    # Remove the plot border (spines)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # Save the plot
    output_file = os.path.join(output_dir,
                               f'cvi_purity_correlation_{os.path.basename(file_path).replace(".csv", "")}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, transparent=True)  # Save with transparent background
    plt.close()

def main(input_folder, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each CSV file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            if '_internal_' in file_name:
                process_csv(file_path, output_dir, 'internal')
            elif '_external_' in file_name:
                process_csv(file_path, output_dir, 'external')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize CVI correlations with purity score using whisker plot for all CSV files in the input folder.')
    parser.add_argument('input_folder', type=str, help='Directory containing the input CSV files')
    parser.add_argument('output_dir', type=str, help='Directory to save the output plots')

    args = parser.parse_args()

    main(args.input_folder, args.output_dir)
