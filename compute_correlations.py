import os
import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings

# Define the CVI metrics
cvi_metrics = [
    'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
    'adjusted_rand_index', 'jaccard_index', 'fowlkes_mallows_index', 'normalized_mutual_info_score',
    'adjusted_mutual_info_score', 'homogeneity_score', 'completeness_score', 'v_measure_score',
    'iCH', 'icSIL', 'iDB', 'iGD43', 'iGD53', 'iWB', 'iPS', 'irCIP', 'iXB'
]

def get_datasets(directory):
    """Return a list of dataset names (file names without extensions) in the given directory."""
    return sorted([os.path.splitext(name)[0] for name in os.listdir(directory) if name.endswith('.csv')])

def read_cvi_data(dir_path, dataset_name):
    """Read the CSV file containing the CVI data for the given dataset from the specified directory."""
    csv_file = f"{dataset_name}.csv"
    file_path = os.path.join(dir_path, csv_file)
    return pd.read_csv(file_path) if os.path.exists(file_path) else None

def preprocess_data(df):
    """Preprocess the DataFrame by dropping columns that are all infs or NaNs and then dropping rows with any infs or NaNs."""
    df = df.replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaN
    df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
    df = df.dropna(axis=0, how='any')  # Drop rows with any NaN values
    return df

def compute_correlation(df, cvi_metric, method='pearson'):
    """Compute the correlation between the given CVI metric and the purity score using the specified method."""
    try:
        if method == 'pearson':
            corr_func = pearsonr
        elif method == 'spearman':
            corr_func = spearmanr
        elif method == 'kendall':
            corr_func = kendalltau
        else:
            raise ValueError(f"Unsupported correlation method: {method}")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            correlation, _ = corr_func(df[cvi_metric], df['purity_score'])
            # Check if any warnings were raised
            for warning in w:
                if "An input array is constant" in str(warning.message):
                    return '--'   # An input array is constant; the correlation coefficient is not defined
            return correlation
    except ValueError:
        return '--'  # x and y must have length at least 2

def main(birch_cvi_dir, dbstream_cvi_dir, stream_kmeans_cvi_dir, output_dir):
    # Define the paths to the directories containing the CVI data for each algorithm
    dirs = {
        "BIRCH": birch_cvi_dir,
        "DBSTREAM": dbstream_cvi_dir,
        "STREAM_KMEANS": stream_kmeans_cvi_dir
    }

    # Get all unique datasets from the three directories
    datasets_birch = set(get_datasets(dirs["BIRCH"]))
    datasets_dbstream = set(get_datasets(dirs["DBSTREAM"]))
    datasets_stream_kmeans = set(get_datasets(dirs["STREAM_KMEANS"]))

    all_datasets = datasets_birch | datasets_dbstream | datasets_stream_kmeans

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionaries to store the correlation results for each method
    correlation_results = {
        'pearson': [],
        'spearman': [],
        'kendall': []
    }

    # Calculate the correlation for each dataset, each CVI metric, and each correlation method
    for dataset in all_datasets:
        for algo, dir_path in dirs.items():
            df = read_cvi_data(dir_path, dataset)
            if df is not None:
                df = preprocess_data(df)
                for cvi in cvi_metrics:
                    for method in correlation_results.keys():
                        correlation = compute_correlation(df, cvi, method=method) if cvi in df.columns else '--'
                        correlation_results[method].append({
                            "CVI": cvi,
                            "Algorithm": algo,
                            "Dataset": dataset,
                            "Correlation": correlation
                        })
            else:
                for cvi in cvi_metrics:
                    for method in correlation_results.keys():
                        correlation_results[method].append({
                            "CVI": cvi,
                            "Algorithm": algo,
                            "Dataset": dataset,
                            "Correlation": '--'
                        })

    # Create and save DataFrames for each correlation method
    for method, records in correlation_results.items():
        correlation_df = pd.DataFrame(records)

        # Replace '--' with np.nan for the pivot operation
        correlation_df['Correlation'] = correlation_df['Correlation'].replace('--', np.nan)

        # Pivot the DataFrame to create the desired table format
        pivot_df = correlation_df.pivot_table(index=["Dataset"], columns=["CVI", "Algorithm"], values="Correlation")

        # Flatten the multi-level columns
        pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

        # Replace np.nan back to '--'
        pivot_df = pivot_df.fillna('--')

        # Ensure all datasets are present in the index
        pivot_df = pivot_df.reindex(all_datasets).fillna('--')

        # Save the results to a CSV file
        output_file = os.path.join(output_dir, f'cvi_purity_correlations_{method}.csv')
        pivot_df.to_csv(output_file)

        print(f"{method.capitalize()} correlation results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute correlations between CVIs and purity score.')
    parser.add_argument('birch_cvi_dir', type=str, help='Directory containing CVI data for BIRCH algorithm')
    parser.add_argument('dbstream_cvi_dir', type=str, help='Directory containing CVI data for DBSTREAM algorithm')
    parser.add_argument('stream_kmeans_cvi_dir', type=str, help='Directory containing CVI data for STREAM_KMEANS algorithm')
    parser.add_argument('output_dir', type=str, help='Directory to save the output tables')

    args = parser.parse_args()

    main(args.birch_cvi_dir, args.dbstream_cvi_dir, args.stream_kmeans_cvi_dir, args.output_dir)
