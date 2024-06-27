import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Define internal and external CVI metrics
internal_cvi_metrics = [
    'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
    'iCH', 'icSIL', 'iDB', 'iGD43', 'iGD53', 'iWB', 'iPS', 'irCIP', 'iXB'
]

external_cvi_metrics = [
    'adjusted_rand_index', 'jaccard_index', 'fowlkes_mallows_index', 'normalized_mutual_info_score',
    'adjusted_mutual_info_score', 'homogeneity_score', 'completeness_score', 'v_measure_score'
]

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
                        "CVI": cvi,
                        "Algorithm": algo,
                        "Value": float(value)  # Ensure value is treated as a float
                    })

    plot_df = pd.DataFrame(plot_data)

    # Ensure that 'CVI' and 'Algorithm' are treated as categorical data
    plot_df['CVI'] = pd.Categorical(plot_df['CVI'], categories=cvi_metrics, ordered=True)
    plot_df['Algorithm'] = plot_df['Algorithm'].astype('category')

    # Set up the plot
    plt.figure(figsize=(8, 9))
    sns.boxplot(x="CVI", y="Value", hue="Algorithm", data=plot_df, palette="Set2", order=cvi_metrics)
    plt.xticks(rotation=0)
    plt.xlabel('CVI', fontsize=14)
    plt.ylabel('Correlation with Purity Score', fontsize=14)

    # Save the plot
    output_file = os.path.join(output_dir,
                               f'cvi_purity_correlation_{os.path.basename(file_path).replace(".csv", "")}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
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
