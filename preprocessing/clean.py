"""
Data Preprocessing Script for CAFUC2 Dataset

Description:
    This script implements the standardized data cleaning pipeline described in the paper.
    It performs two main tasks:
    1. Missing Value Imputation: Fills null values using the median of the column.
    2. Outlier Removal: Detects artifacts using the Interquartile Range (IQR) method
       and replaces them with the mean of adjacent temporal neighbors (t-1 and t+1).

Usage:
    Adjust the INPUT_FOLDER and OUTPUT_FOLDER variables in the __main__ block.
    Run the script: python clean_log.py

Dependencies:
    pandas, numpy
"""

import pandas as pd
import numpy as np
import os


def clean_column(df, column_name, outlier_method='mean_of_neighbors', iqr_multiplier=1.5):
    """
    Cleans a specific column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The dataframe to process.
    - column_name (str): Name of the column to clean.
    - outlier_method (str): Method to handle outliers.
                            'mean_of_neighbors': Replaces outliers with (t-1 + t+1) / 2.
                            'IQR': Clips values to the IQR boundaries.
                            'None': No outlier processing.
    - iqr_multiplier (float): Multiplier for IQR (default 1.5).

    Returns:
    - pd.DataFrame: The dataframe with the processed column.
    """
    df = df.copy()

    # 1. Ensure numeric type, coerce errors to NaN
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    # 2. Missing Value Imputation (Median)
    if df[column_name].notna().any():
        median_value = df[column_name].median()
        # Only fill if there are NaNs to avoid unnecessary operations
        if df[column_name].isna().sum() > 0:
            df[column_name] = df[column_name].fillna(median_value)

    # 3. Outlier Removal
    if outlier_method in ['IQR', 'mean_of_neighbors'] and df[column_name].notna().any():
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        if outlier_method == 'IQR':
            # Simple clipping
            df[column_name] = np.clip(df[column_name], lower_bound, upper_bound)

        elif outlier_method == 'mean_of_neighbors':
            # Create a mask for outliers
            outlier_mask = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)

            if outlier_mask.any():
                # Replace outliers with NaN temporarily to utilize ffill/bfill
                temp_series = df[column_name].where(~outlier_mask, np.nan)

                # Get value at t-1 (forward fill) and t+1 (backward fill)
                forward_fill = temp_series.ffill()
                backward_fill = temp_series.bfill()

                # Handle edge cases (start/end of file)
                forward_fill = forward_fill.fillna(backward_fill)
                backward_fill = backward_fill.fillna(forward_fill)

                # Calculate the mean of neighbors
                average_fill = (forward_fill + backward_fill) / 2

                # Apply substitution only to outlier positions
                df[column_name] = df[column_name].where(~outlier_mask, average_fill)

    return df


def process_folder(input_folder, output_folder, file_type='csv', outlier_method='mean_of_neighbors'):
    """
    Batch processes all compatible files in a directory.
    """
    print(f"Starting processing in folder: '{input_folder}'...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Define supported extensions
    file_extensions = {'csv': ('.csv',), 'excel': ('.xlsx', '.xls')}
    if file_type not in file_extensions:
        print(f"Error: Unsupported file type '{file_type}'. Choose 'csv' or 'excel'.")
        return

    # Iterate through files
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(file_extensions[file_type]):
            file_path = os.path.join(input_folder, filename)
            output_filename = f"cleaned_{os.path.splitext(filename)[0]}.csv"  # Always save as CSV for standardization
            output_path = os.path.join(output_folder, output_filename)

            try:
                print(f"\n--- Processing file: {filename} ---")

                # Load data
                if file_type == 'csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)

                # Process each column
                for col in df.columns:
                    # Skip 'Time' or 'Date' columns from outlier removal to preserve timestamps
                    if 'time' in col.lower() or 'date' in col.lower():
                        print(f"  -> Skipping time/date column: '{col}'")
                        continue

                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(df[col]) or pd.to_numeric(df[col], errors='coerce').notna().any():
                        df = clean_column(df, col, outlier_method=outlier_method)
                    else:
                        print(f"  -> Skipping non-numeric column: '{col}'")

                # Save result
                df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"Successfully saved to: {output_path}")

            except Exception as e:
                print(f"Error processing '{filename}': {e}")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Configuration
    # Note: These paths should be updated by the user before running
    INPUT_FOLDER = './raw_data/'  # Use relative paths for portability
    OUTPUT_FOLDER = './cleaned_data/'
    FILE_TYPE = 'csv'

    # Outlier Method: 'mean_of_neighbors' matches the paper description
    OUTLIER_METHOD = 'mean_of_neighbors'

    # Run pipeline
    process_folder(
        INPUT_FOLDER,
        OUTPUT_FOLDER,
        FILE_TYPE,
        outlier_method=OUTLIER_METHOD
    )