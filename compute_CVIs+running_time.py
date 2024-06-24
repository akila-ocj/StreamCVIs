import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    jaccard_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
)
from pathlib import Path
import argparse
import cvi  # Ensure you have the 'cvi' module installed
from datetime import datetime
import time
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")


def purity_score(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    maxima = np.sum(np.amax(contingency_matrix.values, axis=0))
    return maxima / np.sum(contingency_matrix.sum())


def compute_sklearn_cvis(features, true_labels, predicted_labels):
    results = {}
    times = {}

    # Compute silhouette score for each distance metric
    start_time = time.time()
    try:
        results[f'silhouette_score'] = silhouette_score(features, predicted_labels)
    except ValueError:
        results[f'silhouette_score'] = np.nan
    times[f'silhouette_score'] = time.time() - start_time

    start_time = time.time()
    try:
        results['calinski_harabasz_score'] = calinski_harabasz_score(features, predicted_labels)
    except ValueError:
        results['calinski_harabasz_score'] = np.nan
    times['calinski_harabasz_score'] = time.time() - start_time

    start_time = time.time()
    try:
        results['davies_bouldin_score'] = davies_bouldin_score(features, predicted_labels)
    except ValueError:
        results['davies_bouldin_score'] = np.nan
    times['davies_bouldin_score'] = time.time() - start_time

    start_time = time.time()
    try:
        results['adjusted_rand_index'] = adjusted_rand_score(true_labels, predicted_labels)
    except ValueError:
        results['adjusted_rand_index'] = np.nan
    times['adjusted_rand_index'] = time.time() - start_time

    start_time = time.time()
    try:
        results[f'jaccard_index'] = jaccard_score(true_labels, predicted_labels, average='macro')
    except ValueError:
        results[f'jaccard_index'] = np.nan
    times[f'jaccard_index'] = time.time() - start_time

    start_time = time.time()
    try:
        results['fowlkes_mallows_index'] = fowlkes_mallows_score(true_labels, predicted_labels)
    except ValueError:
        results['fowlkes_mallows_index'] = np.nan
    times['fowlkes_mallows_index'] = time.time() - start_time

    start_time = time.time()
    try:
        results[f'normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, predicted_labels)
    except ValueError:
        results[f'normalized_mutual_info_score'] = np.nan
    times[f'normalized_mutual_info_score'] = time.time() - start_time

    start_time = time.time()
    try:
        results[f'adjusted_mutual_info_score'] = adjusted_mutual_info_score(true_labels, predicted_labels)
    except ValueError:
        results[f'adjusted_mutual_info_score'] = np.nan
    times[f'adjusted_mutual_info_score'] = time.time() - start_time

    start_time = time.time()
    try:
        results['homogeneity_score'] = homogeneity_score(true_labels, predicted_labels)
    except ValueError:
        results['homogeneity_score'] = np.nan
    times['homogeneity_score'] = time.time() - start_time

    start_time = time.time()
    try:
        results['completeness_score'] = completeness_score(true_labels, predicted_labels)
    except ValueError:
        results['completeness_score'] = np.nan
    times['completeness_score'] = time.time() - start_time

    start_time = time.time()
    try:
        results['v_measure_score'] = v_measure_score(true_labels, predicted_labels)
    except ValueError:
        results['v_measure_score'] = np.nan
    times['v_measure_score'] = time.time() - start_time

    start_time = time.time()
    try:
        results['purity_score'] = purity_score(true_labels, predicted_labels)
    except ValueError:
        results['purity_score'] = np.nan
    times['purity_score'] = time.time() - start_time

    return results, times


def compute_incremental_cvis(features, labels):
    results = {}
    times = {}
    incremental_cvis = {
        'iCH': cvi.CH(),
        'icSIL': cvi.cSIL(),
        'iDB': cvi.DB(),
        'iGD43': cvi.GD43(),
        'iGD53': cvi.GD53(),
        'iWB': cvi.WB(),
        'iPS': cvi.PS(),
        'irCIP': cvi.rCIP(),
        'iXB': cvi.XB()
    }
    incremental_crit = {name: np.zeros(len(labels)) for name in incremental_cvis.keys()}

    for ix in range(len(labels)):
        for cvi_name, cvi_instance in incremental_cvis.items():
            start_time = time.time()
            try:
                incremental_crit[cvi_name][ix] = cvi_instance.get_cvi(features[ix, :], labels[ix])
            except (ValueError, ZeroDivisionError):
                incremental_crit[cvi_name][ix] = np.nan
            times[cvi_name] = time.time() - start_time

    for cvi_name in incremental_cvis.keys():
        try:
            results[cvi_name] = np.nanmean(incremental_crit[cvi_name])
        except (ValueError, ZeroDivisionError):
            results[cvi_name] = np.nan

    return results, times


def process_dataset(file_path, output_folder_cvis, output_folder_time):
    # Read the CSV file
    data = pd.read_csv(file_path)
    data = data.dropna()

    all_incremental_results = []
    all_times = []

    for i in range(1, len(data) + 1):
        incremental_data = data.iloc[:i]
        features = incremental_data.iloc[:, :-2].values
        true_labels = incremental_data['TrueLabel']
        predicted_labels = incremental_data['PredictedLabel']

        sklearn_cvis, sklearn_times = compute_sklearn_cvis(features, true_labels, predicted_labels)
        incremental_cvis, incremental_times = compute_incremental_cvis(features, predicted_labels)

        combined_results = {**sklearn_cvis, **incremental_cvis}
        combined_results['Iteration'] = i

        combined_times = {**sklearn_times, **incremental_times}
        combined_times['Iteration'] = i

        result_row = incremental_data.iloc[-1].to_dict()
        result_row.update(combined_results)

        time_row = incremental_data.iloc[-1].to_dict()
        time_row.update(combined_times)

        all_incremental_results.append(result_row)
        all_times.append(time_row)

    incremental_results_df = pd.DataFrame(all_incremental_results)
    times_df = pd.DataFrame(all_times)

    output_directory_cvis = Path(output_folder_cvis)
    output_directory_time = Path(output_folder_time)
    output_directory_cvis.mkdir(parents=True, exist_ok=True)
    output_directory_time.mkdir(parents=True, exist_ok=True)

    cvis_file_path = output_directory_cvis / f'{Path(file_path).stem}_cvis.csv'
    time_file_path = output_directory_time / f'{Path(file_path).stem}_times.csv'

    incremental_results_df.to_csv(cvis_file_path, index=False)
    times_df.to_csv(time_file_path, index=False)

    print(f"Incremental results saved to {cvis_file_path}")
    print(f"Running times saved to {time_file_path}")
    return cvis_file_path, time_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute CVIs and their running times for each CSV in the input folder.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('output_folder_cvis', type=str, help='Path to the output folder to save computed CVIs.')
    parser.add_argument('output_folder_time', type=str,
                        help='Path to the output folder to save running times for CVIs.')
    args = parser.parse_args()

    csv_file = Path(args.csv_file)
    output_folder_cvis = Path(args.output_folder_cvis)
    output_folder_time = Path(args.output_folder_time)

    process_dataset(csv_file, output_folder_cvis, output_folder_time)
