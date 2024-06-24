import os
import pandas as pd
import numpy as np
import random
from river import cluster
from sklearn.metrics import accuracy_score
from pathlib import Path

def purity_score(y_true, y_pred):
    # Compute contingency matrix (also called confusion matrix)
    contingency_matrix = pd.crosstab(y_true, y_pred)
    # Return purity
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

def load_data(file_path):
    # Load the data from a CSV file
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-2].values
    y = data.iloc[:, -2].values
    set_type = data.iloc[:, -1].values
    return X, y, set_type

def main(input_folder, output_folder):
    # Get list of all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    # Load the first CSV file for training and validation
    first_csv_path = os.path.join(input_folder, csv_files[0])
    X, y, set_type = load_data(first_csv_path)

    X_train, y_train = X[set_type == 'train'], y[set_type == 'train']
    X_val, y_val = X[set_type == 'validate'], y[set_type == 'validate']

    best_purity = -np.inf
    best_purity2 = -np.inf
    best_params = None
    all_params = []
    all_purities = []

    n_clusters = len(np.unique(y_train))

    for _ in range(200):
        chosen_params = {
            'n_clusters': n_clusters,
            'chunk_size': random.choice([5, 10, 20, 30, 50, 100, 200]),
            'halflife': random.choice([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0]),
            'mu': random.choice([-2, -1, 0, 1, 2, 3, 4]),
            'sigma': random.choice([0.1, 0.5, 1, 2, 3, 5, 10]),
            'p': random.choice([1, 1.5, 2, 2.5, 3, 4])
        }
        model = cluster.STREAMKMeans(**chosen_params)
        for xi in X_train:
            model.learn_one(dict(enumerate(xi)))

        y_pred_train = [model.predict_one(dict(enumerate(xi))) for xi in X_train]
        y_pred_val = [model.predict_one(dict(enumerate(xi))) for xi in X_val]

        pred_clusters = len(np.unique(y_pred_val))
        if pred_clusters == 1 or pred_clusters >= 2 * n_clusters:
            continue

        purity = purity_score(y_val, y_pred_val)
        all_params.append(chosen_params)
        all_purities.append(purity)

        if purity > best_purity:
            best_purity = purity
            best_params = chosen_params

    root_directory = Path(os.getcwd())
    output_directory = root_directory / output_folder
    output_directory.mkdir(exist_ok=True)

    for test_csv in csv_files:
        X_test, y_test, set_type = load_data(os.path.join(input_folder, test_csv))
        X_test_set = X_test[set_type == 'test']
        y_test_set = y_test[set_type == 'test']

        model = cluster.STREAMKMeans(**best_params)
        for xi in X_train:
            model.learn_one(dict(enumerate(xi)))

        y_pred_test = [model.predict_one(dict(enumerate(xi))) for xi in X_test_set]

        output_path = output_directory / f'{test_csv}_predicted.csv'
        test_results_df = pd.DataFrame(X_test_set, columns=[f'Feature_{i}' for i in range(X_test_set.shape[1])])
        test_results_df['TrueLabel'] = y_test_set
        test_results_df['PredictedLabel'] = y_pred_test
        test_results_df.to_csv(output_path, index=False)

    if best_params is not None:
        best_params_path = output_directory / 'best_params.csv'
        best_results_df = pd.DataFrame([{**best_params, 'purity': best_purity}])
        best_results_df.to_csv(best_params_path, index=False)

        all_params_path = output_directory / 'all_params.csv'
        all_results_df = pd.DataFrame(all_params)
        all_results_df['purity'] = all_purities
        all_results_df.to_csv(all_params_path, index=False)
    else:
        print("No valid clustering parameters found.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train StreamKMeans and predict labels on test sets.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing CSV files.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to save results.')
    args = parser.parse_args()

    main(args.input_folder, args.output_folder)
