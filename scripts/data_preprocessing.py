"""
Script to preprocess football match data from multiple CSV files.

Usage:
------
Run this script from the terminal as follows, from the root directory of the project:

    python data_preprocessing.py --input_dir data/raw --output_dir data/processed --num_features 20

Parameters:
-----------
input_dir : str
    Path to the folder containing the CSV files to be processed.
output_dir : str
    Directory where the processed CSV files will be saved.
num_features : int
    Number of top features to select using the mRMR feature selection method.

This script will read each CSV file in the input folder, perform feature engineering,
select relevant features while addressing feature correlation, handle missing values,
and save the processed data to the specified output directory.
"""

import os
import argparse
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from mrmr import mrmr_classif

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess football match data from CSV files.")
    parser.add_argument("--input_dir", type=str, help="Path to the folder containing the CSV files to be processed.")
    parser.add_argument("--output_dir", type=str, help="Directory where the processed CSV files will be saved.")
    parser.add_argument("--num_features", type=int, help="Number of top features to select using mRMR.")

    return parser.parse_args()

def load_csv_files(input_folder):
    """
    Load all CSV files from the specified input folder.

    Parameters:
    input_folder (str): Path to the folder containing the CSV files.

    Returns:
    list of tuples: A list where each tuple contains the filename and the corresponding DataFrame.
    """
    data_files = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            data = pd.read_csv(file_path)
            data_files.append((filename, data))
    return data_files

def feature_engineering(df):
    """
    Perform feature engineering on the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: The DataFrame with new features added.
    """
    df["Over2.5"] = np.where(df["FTHG"] + df["FTAG"] > 2, 1, 0)
    df['AvgHomeGoalsScored'] = df.groupby('HomeTeam')['FTHG'].transform('mean').round(2)
    df['AvgAwayGoalsScored'] = df.groupby('AwayTeam')['FTAG'].transform('mean').round(2)
    df['AvgHomeGoalsConceded'] = df.groupby('HomeTeam')['FTAG'].transform('mean').round(2)
    df['AvgAwayGoalsConceded'] = df.groupby('AwayTeam')['FTHG'].transform('mean').round(2)
    df['HomeOver2.5Perc'] = df.groupby('HomeTeam')['Over2.5'].transform('mean').round(2)
    df['AwayOver2.5Perc'] = df.groupby('AwayTeam')['Over2.5'].transform('mean').round(2)

    df = df.sort_values(by=['HomeTeam', 'Date'])
    df['Last5HomeGoalsScored'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.rolling(5, min_periods=1).mean()).round(2)
    df['Last5HomeOver2.5Count'] = df.groupby('HomeTeam')['Over2.5'].transform(lambda x: x.rolling(5, min_periods=1).sum()).round(2)

    df = df.sort_values(by=['AwayTeam', 'Date'])
    df['Last5AwayGoalsScored'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.rolling(5, min_periods=1).mean()).round(2)
    df['Last5AwayOver2.5Count'] = df.groupby('AwayTeam')['Over2.5'].transform(lambda x: x.rolling(5, min_periods=1).sum()).round(2)
    
    return df

def drop_useless_columns(df, columns_to_drop):
    """
    Drop the specified columns from the DataFrame if they exist.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    columns_to_drop (list of str): List of column names to drop.

    Returns:
    pd.DataFrame: The DataFrame with specified columns dropped.
    """
    for column in columns_to_drop:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)
        else:
            print(f"Column {column} not found in the dataframe")

    return df

def feature_selection(df, target_column="Over2.5", num_features=20, clustering_threshold=0.7):
    """
    Perform feature selection using mRMR and hierarchical clustering.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    target_column (str): The target variable column name.
    num_features (int): The number of features to select using mRMR.

    Returns:
    list: A list of selected feature names after clustering.
    """
    numerical_columns = df.select_dtypes(exclude='object').columns.tolist()
    X = df[numerical_columns].drop([target_column], axis=1)
    y = df[target_column]
    
    selected_features = mrmr_classif(X=X, y=y, K=num_features)
    corr_matrix = df[selected_features].corr(method='spearman')
    
    dist = sch.distance.pdist(corr_matrix)
    linkage = sch.linkage(dist, method='average')
    cluster_ids = sch.fcluster(linkage, clustering_threshold, criterion='distance')
    
    selected_features_clustered = []
    for cluster_id in pd.Series(cluster_ids).unique():
        cluster_features = corr_matrix.columns[pd.Series(cluster_ids) == cluster_id]
        selected_features_clustered.append(cluster_features[0])

    return selected_features_clustered

def handle_missing_values(df, missing_threshold=10):
    """
    Handle missing values by removing columns and rows with excessive missing data.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    missing_threshold (int): The maximum allowed count of missing values per column before dropping the column.

    Returns:
    pd.DataFrame: The cleaned DataFrame with missing values handled.
    """
    missing_values_count = df.isnull().sum()
    print("Missing values in each column:\n", missing_values_count)
    
    columns_to_drop = missing_values_count[missing_values_count > missing_threshold].index
    print("\nColumns to drop due to excessive missing values:\n", columns_to_drop)
    
    df = df.drop(columns=columns_to_drop)
    df = df.dropna()
    print("\nRemaining missing values:\n", df.isnull().sum())
    
    return df

def save_preprocessed_data(df, output_folder, filename):
    """
    Save the preprocessed DataFrame to the specified output folder with a modified filename.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    output_folder (str): Path to the folder where the processed CSV file will be saved.
    filename (str): The original filename of the CSV file.

    Returns:
    None
    """
    output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_preprocessed.csv")
    df.to_csv(output_file_path, index=False)
    print(f"Preprocessed file saved as {output_file_path}\n")

def preprocess_and_save_csv(input_folder, output_folder, num_features, missing_threshold=10, clustering_threshold=0.7):
    """
    Preprocess CSV files in the specified input folder and save the processed files to the output folder.

    Parameters:
    input_folder (str): Path to the folder containing the CSV files to be processed.
    output_folder (str): Path to the folder where the processed CSV files will be saved.
    num_features (int): The number of top features to select using mRMR.
    missing_threshold (int): The maximum allowed count of missing values per column before dropping the column.
    clustering_threshold (float): The threshold for hierarchical clustering to form flat clusters.

    Returns:
    None
    """
    data_files = load_csv_files(input_folder)

    for filename, df in data_files:
        print(f"Processing {filename}...")
        
        # Feature Engineering
        df = feature_engineering(df)
        print("Feature engineering completed.")

        # Drop useless columns
        df = drop_useless_columns(df, ['HTHG', 'HTAG'])
        print("Useless columns dropped.")

        # Feature Selection
        selected_features = feature_selection(df, num_features=num_features, clustering_threshold=clustering_threshold)
        print("Selected features after clustering:", selected_features)
        
        # Create final dataframe with selected features and handle missing values
        categorical_columns = df.select_dtypes(include='object').columns.tolist()
        df_selected = df[categorical_columns + selected_features + ['Over2.5']]
        df_selected = handle_missing_values(df_selected, missing_threshold)

        # Save the preprocessed dataframe
        save_preprocessed_data(df_selected, output_folder, filename)

if __name__ == "__main__":
    """
    Example usage:
    python preprocess_football_data.py <input_folder> <output_folder> <num_features>
    
    Arguments:
    <input_folder>: The folder containing the CSV files to be processed.
    <output_folder>: The folder where the processed CSV files will be saved.
    <num_features>: The number of top features to select using mRMR.
    """

    args = parse_arguments()
    
    # check if the input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist.")
        exit(1)

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preprocess_and_save_csv(args.input_dir, args.output_dir, args.num_features)
