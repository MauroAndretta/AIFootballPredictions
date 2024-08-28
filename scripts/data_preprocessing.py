"""
Script to preprocess football match data from multiple CSV files.

Usage:
------
Run this script from the terminal as follows, from the root directory of the project:

    python scripts/data_preprocessing.py --raw_data_input_dir data/raw --processed_data_output_dir data/processed --num_features 20 --clustering_threshold 0.5

Parameters:
-----------
raw_data_input_dir : str
    Path to the folder containing the CSV files to be processed.
processed_data_output_dir : str
    Directory where the processed CSV files will be saved.
num_features : int
    Number of top features to select using the mRMR feature selection method.
clustering_threshold : float
    The threshold for hierarchical clustering to form flat clusters.

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
from sklearn.preprocessing import StandardScaler

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess football match data from CSV files.")
    parser.add_argument("--raw_data_input_dir", required=True, type=str, help="Path to the folder containing the CSV files to be processed.")
    parser.add_argument("--processed_data_output_dir", required=True, type=str, help="Directory where the processed CSV files will be saved.")
    parser.add_argument("--num_features", type=int, default=20, help="Number of top features to select using mRMR.")
    parser.add_argument("--clustering_threshold", type=float, default=0.5, help="The threshold for hierarchical clustering to form flat clusters.")

    return parser.parse_args()

def load_csv_files(input_folder: str) -> list:
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

def determine_season(date: pd.Timestamp) -> str:
    """
    Determine the season based on the date of the match.

    Parameters:
    date (datetime): The date of the match.

    Returns:
    str: The season in the format "YYYY/YYYY".
    """

    year = date.year
    if date.month >= 8:  # Assuming the season starts in August
        return f"{year}/{year + 1}"
    else:
        return f"{year - 1}/{year}"

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: The DataFrame with new features added.
    """
    # Convert Date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    # Create a new column Season based on the Date
    df['Season'] = df['Date'].apply(determine_season)

    # create the target variable Over2.5 
    df["Over2.5"] = np.where(df["FTHG"] + df["FTAG"] > 2, 1, 0)
    # Group by HomeTeam and calculate the average Full Time Home Goals
    df['AvgHomeGoalsScored'] = df.groupby(['Season', 'HomeTeam'])['FTHG'].transform('mean').round(2)
    # Group by AwayTeam and calculate the average Full Time Away Goals
    df['AvgAwayGoalsScored'] = df.groupby(['Season', 'AwayTeam'])['FTAG'].transform('mean').round(2)
    # Group by HomeTeam and calculate the average Full Time Away Goals (which are the goals conceded by HomeTeam)
    df['AvgHomeGoalsConceded'] = df.groupby(['Season', 'HomeTeam'])['FTAG'].transform('mean').round(2)
    # Group by AwayTeam and calculate the average Full Time Home Goals (which are the goals conceded by AwayTeam)
    df['AvgAwayGoalsConceded'] = df.groupby(['Season', 'AwayTeam'])['FTHG'].transform('mean').round(2)
    # Group by HomeTeam and calculate the percentage of games with Over 2.5 goals
    df['HomeOver2.5Perc'] = (df.groupby(['Season', 'HomeTeam'])['Over2.5'].transform('mean') * 100).round(2)
    # Group by HomeTeam and calculate the percentage of games with Over 2.5 goals
    df['AwayOver2.5Perc'] = (df.groupby(['Season', 'AwayTeam'])['Over2.5'].transform('mean') * 100).round(2)

    # Sort the dataframe by HomeTeam and Date
    df = df.sort_values(by=['HomeTeam', 'Date'])
    # Create a rolling average of the last 5 games for the Full Time Home Goals
    df['AvgLast5HomeGoalsScored'] = df.groupby(['Season', 'HomeTeam'])['FTHG'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()).round(2)
    df['AvgLast5HomeGoalsConceded'] = df.groupby(['Season', 'HomeTeam'])['FTAG'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()).round(2)
    # Create a rolling sum of the last 5 games for Over 2.5 goals for home matches
    df['Last5HomeOver2.5Count'] = df.groupby(['Season', 'HomeTeam'])['Over2.5'].transform(
        lambda x: x.rolling(5, min_periods=1).sum()).round(2)
    # Calculate the percentage of Over 2.5 goals in the last 5 home matches
    df['Last5HomeOver2.5Perc'] = df.groupby(['Season', 'HomeTeam'])['Over2.5'].transform(
        lambda x: x.rolling(5, min_periods=1).mean() * 100).round(2)

    # Sort the dataframe by HomeTeam and Date
    df = df.sort_values(by=['AwayTeam', 'Date'])
    # Create a rolling average of the last 5 games for the Full Time Away Goals
    df['AvgLast5AwayGoalsScored'] = df.groupby(['Season', 'AwayTeam'])['FTAG'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()).round(2)
    df['AvgLast5AwayGoalsConceded'] = df.groupby(['Season', 'AwayTeam'])['FTHG'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()).round(2)
    # Create a rolling sum of the last 5 games for Over 2.5 goals for away matches
    df['Last5AwayOver2.5Count'] = df.groupby(['Season', 'AwayTeam'])['Over2.5'].transform(
        lambda x: x.rolling(5, min_periods=1).sum()).round(2)
    # Create a rolling sum of the last 5 games for Over 2.5 goals for away matches
    df['Last5AwayOver2.5Perc'] = df.groupby(['Season', 'AwayTeam'])['Over2.5'].transform(
        lambda x: x.rolling(5, min_periods=1).mean() * 100).round(2)    
    return df


def drop_useless_columns(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
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

def feature_selection(df, target_column="Over2.5", num_features=20, clustering_threshold=0.5):
    """
    Perform feature selection using mRMR and hierarchical clustering.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    target_column (str): The target variable column name.
    num_features (int): The number of features to select using mRMR.
    clustering_threshold (float): The threshold for hierarchical clustering to form flat clusters.

    Returns:
    list: A list of selected feature names after clustering.
    """
    try:
        numerical_columns = df.drop(["Date"], axis=1).select_dtypes(exclude='object').columns.tolist()
        X = df[numerical_columns].drop([target_column], axis=1)
        y = df[target_column]
        
        # 1.0- Select the top features using mRMR
        selected_features = mrmr_classif(X=X, y=y, K=num_features)

        """
        DRASTIC DECREASE IN PERFORMANCE WHEN STANDARDIZING THE FEATURES
        # 2.0- Standardize the selected features to help with clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[selected_features]) 

            # 2.1- Create a DataFrame with the scaled features
        df_scaled = pd.DataFrame(scaled_features, columns=selected_features)
        #replace the original data with the scaled data
        df[selected_features] = df_scaled[selected_features]
        """

        # 3.0- Perform hierarchical clustering to group correlated features

            # 3.1- Calculate the Spearman correlation matrix
        corr_matrix = df[selected_features].corr(method='spearman')
        
            # 3.2- Perform hierarchical clustering d = 1 - r
        dist = sch.distance.pdist(corr_matrix, metric='euclidean')
        linkage = sch.linkage(dist, method='average')
        cluster_ids = sch.fcluster(linkage, clustering_threshold, criterion='distance')
        
        # 4.0- Select the feature with the highest variance within each cluster
        selected_features_clustered = []
        for cluster_id in pd.Series(cluster_ids).unique():
            cluster_features = corr_matrix.columns[pd.Series(cluster_ids) == cluster_id]
            # Select the feature with the highest variance
            highest_variance_feature = cluster_features[np.argmax(df[cluster_features].var())]
            selected_features_clustered.append(highest_variance_feature)

        return selected_features_clustered
    
    except Exception as e:
        print(f"Error during feature selection: {e}")
        return []

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

def preprocess_and_save_csv(input_folder, output_folder, num_features, missing_threshold=10, clustering_threshold = 0.5):
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
        # All the features related to the goals scored in a match, are higly biasing for the model, so we can drop them
        df = drop_useless_columns(df, ['FTHG', 'FTAG', 'HTHG', 'HTAG'] )
        print("Useless columns dropped.")

        # Handle missing values
        df = handle_missing_values(df, missing_threshold=missing_threshold)
        print("Missing values handled.")

        # Feature Selection
        selected_features = feature_selection(df, num_features=num_features, clustering_threshold=clustering_threshold)
        print(f"Number of selected features: {len(selected_features)}")
        print("Selected features after clustering:", selected_features)
        
        # Create final dataframe with selected features
        categorical_columns = df.select_dtypes(include='object').columns.tolist()
        df_selected = df[["Date"] + categorical_columns + selected_features + ['Over2.5']]

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
    <clustering_threshold>: The threshold for hierarchical clustering to form flat clusters.
    """

    args = parse_arguments()
    
    # check if the input directory exists
    if not os.path.exists(args.raw_data_input_dir):
        print(f"Input directory {args.raw_data_input_dir} does not exist.")
        exit(1)

    # create the output directory if it does not exist
    if not os.path.exists(args.processed_data_output_dir):
        os.makedirs(args.processed_data_output_dir)

    preprocess_and_save_csv(args.raw_data_input_dir, args.processed_data_output_dir, args.num_features, args.clustering_threshold)
