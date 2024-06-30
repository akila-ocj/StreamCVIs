import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Define the CVI metrics in the desired order
cvi_metrics = ['adjusted_rand_index', 'jaccard_index', 'fowlkes_mallows_index', 'homogeneity_score',
               'normalized_mutual_info_score', 'v_measure_score']

# Mapping of CVI metrics to their acronyms
cvi_acronyms = {
    'davies_bouldin_score': 'DB',
    'silhouette_score': 'Sil',
    'adjusted_rand_index': 'ARI',
    'jaccard_index': 'Jaccard',
    'fowlkes_mallows_index': 'FMI',
    'homogeneity_score': 'Homogeneity',
    'normalized_mutual_info_score': 'NMI',
    'v_measure_score': 'V-measure',
    'calinski_harabasz_score': 'CH',
    'adjusted_mutual_info_score': 'AMI'
}

def read_csv_files(directory):
    """Read all CSV files in the given directory and return a concatenated DataFrame."""
    csv_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.csv')]
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['Dataset'] = os.path.basename(file)  # Add a column to indicate which file the data comes from
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def remove_outliers(df):
    """Remove outliers from the DataFrame using the IQR method."""
    for cvi in cvi_metrics:
        if (cvi in df.columns) and (not df[cvi].isnull().all()):  # Check if column exists and is not all NaNs
            Q1 = df[cvi].quantile(0.25)
            Q3 = df[cvi].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[cvi] >= lower_bound) & (df[cvi] <= upper_bound)]
    return df

def process_data(df):
    """Process the data for plotting."""
    plot_data = []

    for cvi in cvi_metrics:
        if cvi in df.columns:
            # Drop rows with np.nan values in the specific column
            df_clean = df.dropna(subset=[cvi])
            for value, dataset in zip(df_clean[cvi], df_clean['Dataset']):
                plot_data.append({
                    "CVI": cvi,
                    "Value": float(value),  # Ensure value is treated as a float
                    "Dataset": dataset
                })

    return pd.DataFrame(plot_data)

def generate_whisker_plot(data, output_dir):
    """Generate a whisker plot for all CVIs and save it in the output directory."""
    # Ensure that 'CVI' is treated as categorical data
    data['CVI'] = pd.Categorical(data['CVI'], categories=cvi_metrics, ordered=True)

    # Map CVI names to acronyms for plot labels
    data['CVI_label'] = data['CVI'].map(lambda x: cvi_acronyms.get(x, x))

    # Set up the plot
    plt.figure(figsize=(4, 3))
    sns.boxplot(x="CVI_label", y="Value", data=data, palette="Set2",
                order=[cvi_acronyms.get(x, x) for x in cvi_metrics])
    plt.xticks(rotation=0, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('CVI', fontsize=0)
    plt.ylabel('Time (Seconds)', fontsize=6)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_file = os.path.join(output_dir, 'combined_cvi_whisker_plot.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

def generate_bar_chart(data, output_dir, statistic='mean'):
    """Generate a bar chart for the mean or median of all CVIs and save it in the output directory."""
    # Calculate the statistic
    if statistic == 'mean':
        cvi_stat = data.groupby('CVI')['Value'].mean()
    elif statistic == 'median':
        cvi_stat = data.groupby('CVI')['Value'].median()
    else:
        raise ValueError("Statistic must be either 'mean' or 'median'")

    # Convert to DataFrame for plotting
    cvi_stat_df = cvi_stat.reset_index()

    # Map CVI names to acronyms for plot labels
    cvi_stat_df['CVI_label'] = cvi_stat_df['CVI'].map(lambda x: cvi_acronyms.get(x, x))

    # Set up the plot
    plt.figure(figsize=(4, 3))
    sns.barplot(x='CVI_label', y='Value', data=cvi_stat_df, palette="Set2",
                order=[cvi_acronyms.get(x, x) for x in cvi_metrics])
    plt.xticks(rotation=0, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('CVI', fontsize=0)
    plt.ylabel('Time (Seconds)', fontsize=6)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot
    output_file = os.path.join(output_dir, f'combined_cvi_{statistic}_bar_chart.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

def main(birch_dir, dbstream_dir, stream_kmeans_dir, output_dir):
    # Read CSV files from each directory
    birch_data = read_csv_files(birch_dir)
    dbstream_data = read_csv_files(dbstream_dir)
    stream_kmeans_data = read_csv_files(stream_kmeans_dir)

    # Combine all data into a single DataFrame
    combined_data = pd.concat([birch_data, dbstream_data, stream_kmeans_data], ignore_index=True)

    # Remove outliers
    combined_data = remove_outliers(combined_data)

    # Process data for plotting
    plot_data = process_data(combined_data)

    # Generate combined whisker plot
    generate_whisker_plot(plot_data, output_dir)

    # Generate bar charts for mean and median
    generate_bar_chart(plot_data, output_dir, statistic='mean')
    generate_bar_chart(plot_data, output_dir, statistic='median')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate whisker plots, bar charts, heatmaps, and violin plots for CVIs from different algorithms.')
    parser.add_argument('birch_dir', type=str, help='Directory containing datasets for BIRCH algorithm')
    parser.add_argument('dbstream_dir', type=str, help='Directory containing datasets for DBSTREAM algorithm')
    parser.add_argument('stream_kmeans_dir', type=str, help='Directory containing datasets for STREAM_KMEANS algorithm')
    parser.add_argument('output_dir', type=str, help='Directory to save the output plots')

    args = parser.parse_args()

    main(args.birch_dir, args.dbstream_dir, args.stream_kmeans_dir, args.output_dir)
