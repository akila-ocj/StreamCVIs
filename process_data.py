import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse


def process_file(input_file, output_folder):
    # Read CSV file into a DataFrame
    df = pd.read_csv(input_file, header=None)

    # Separate features and labels
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Convert scaled features back to DataFrame
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

    # Combine scaled features and labels back into one DataFrame
    df_scaled = pd.concat([features_scaled, labels], axis=1)

    # Shuffle the data
    df_shuffled = shuffle(df_scaled, random_state=42)

    # Split the data into train (70%) and test (30%)
    train_validate, test = train_test_split(df_shuffled, test_size=0.3, random_state=42)

    # Split the train data into train (80%) and validate (20%)
    train, validate = train_test_split(train_validate, test_size=0.2, random_state=42)

    # Add the dataset labels
    train['dataset'] = 'train'
    validate['dataset'] = 'validate'
    test['dataset'] = 'test'

    # Concatenate all parts
    final_df = pd.concat([train, validate, test])

    # Prepare output file path
    output_file_path = os.path.join(output_folder, os.path.basename(input_file))

    # Save the processed data to the output folder
    final_df.to_csv(output_file_path, index=False, header=False)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Normalize, shuffle, split into train/validate/test, and label a single CSV data file.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('output_folder', type=str,
                        help='Path to the output folder where the processed file will be saved.')

    args = parser.parse_args()

    # Process the file
    process_file(args.input_file, args.output_folder)
