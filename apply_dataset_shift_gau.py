import os
import pandas as pd
import numpy as np
import argparse


def apply_gaussian_noise(data, noise_level):
    noisy_data = data + np.random.normal(0, noise_level, data.shape)
    return noisy_data


def process_file(input_file, output_folder, frac):
    # Read CSV file into a DataFrame
    df = pd.read_csv(input_file, header=None)

    # Identify test data
    test_data = df[df.iloc[:, -1] == 'test']

    # Select a fraction of the test data to apply Gaussian noise
    test_sample = test_data.sample(frac=frac, random_state=42)
    rest_test = test_data.drop(test_sample.index)

    # Levels of Gaussian noise
    noise_levels = {'mild': 0.05, 'moderate': 0.5, 'severe': 0.9}

    for level, noise_level in noise_levels.items():
        # Apply Gaussian noise to the selected sample
        noisy_test_sample = test_sample.copy()
        noisy_test_sample.iloc[:, :-2] = apply_gaussian_noise(noisy_test_sample.iloc[:, :-2], noise_level)

        # Combine the noisy sample with the rest of the test data
        new_test_data = pd.concat([rest_test, noisy_test_sample])

        # Combine with the rest of the dataset (train and validate)
        train_validate_data = df[df.iloc[:, -1] != 'test']
        final_df = pd.concat([train_validate_data, new_test_data])

        # Prepare output file path
        output_file_path = os.path.join(output_folder,
                                        f'{os.path.splitext(os.path.basename(input_file))[0]}_{level}_gaussian-noise.csv')

        # Save the processed data to the output folder
        final_df.to_csv(output_file_path, index=False, header=False)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Apply different levels of Gaussian noise to a fraction of the test data.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where the processed files will be saved.')
    parser.add_argument('--frac', type=float, default=0.1, help='Fraction of the test data to apply Gaussian noise (default is 0.1).')

    args = parser.parse_args()

    # Process the file
    process_file(args.input_file, args.output_folder, args.frac)
