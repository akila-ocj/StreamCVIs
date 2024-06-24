import os
import pandas as pd
import argparse


def remove_class_instances(df, severity, fraction, class_fraction):
    # Identify test data
    test_data = df[df.iloc[:, -1] == 'test']
    class_labels = sorted(test_data.iloc[:, -2].unique())

    if class_fraction == 'one':
        # Select the first class
        classes_to_remove = [class_labels[0]]
    elif class_fraction == 'half':
        # Select the first half of the classes
        classes_to_remove = class_labels[:len(class_labels) // 2]
    else:
        raise ValueError("class_fraction must be either 'one' or 'half'")

    reduced_test_data = test_data.copy()

    # Remove a fraction of the instances of the selected classes
    for class_label in classes_to_remove:
        class_data = test_data[test_data.iloc[:, -2] == class_label]
        class_sample = class_data.sample(frac=fraction, random_state=42)
        reduced_test_data = reduced_test_data.drop(class_sample.index)

    # Combine with the rest of the dataset (train and validate)
    train_validate_data = df[df.iloc[:, -1] != 'test']
    final_df = pd.concat([train_validate_data, reduced_test_data])

    return final_df


def process_file(input_file, output_folder, class_fraction):
    # Read CSV file into a DataFrame
    df = pd.read_csv(input_file, header=None)

    # Define severity levels and corresponding fractions to remove
    severity_levels = {
        'mild': 0.05,
        'moderate': 0.5,
        'severe': 0.9
    }

    for severity, fraction in severity_levels.items():
        # Process the file to remove instances based on severity level
        final_df = remove_class_instances(df, severity, fraction, class_fraction)

        # Prepare output file path
        output_file_path = os.path.join(output_folder,
                                        f'{os.path.splitext(os.path.basename(input_file))[0]}_{severity}_knock-out.csv')

        # Save the processed data to the output folder
        final_df.to_csv(output_file_path, index=False, header=False)


if __name__ == "__main__":
    # Set up argument parsing
    # Example usage: python script.py /path/to/input/file.csv /path/to/output/folder one
    # Example usage: python script.py /path/to/input/file.csv /path/to/output/folder half
    parser = argparse.ArgumentParser(
        description='Remove varying percentages of class instances in the test set based on severity levels and class fraction.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file.')
    parser.add_argument('output_folder', type=str,
                        help='Path to the output folder where the processed files will be saved.')
    parser.add_argument('class_fraction', type=str, choices=['one', 'half'],
                        help='Fraction of classes to remove ("one" or "half").')

    args = parser.parse_args()

    # Process the file
    process_file(args.input_file, args.output_folder, args.class_fraction)
