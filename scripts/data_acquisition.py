"""
Script to gather and merge football match data from multiple leagues and seasons.

Usage:
------
Run this script from the terminal in the root folder as follows:

    python scripts/download_football_data.py --leagues E0 I1 SP1 F1 D1 --seasons 2324 2223 --output_dir data/raw

Parameters:
-----------
--leagues : str
    A space-separated list of league acronyms (e.g., E0 I1 SP1).
--seasons : str
    A space-separated list of season codes (e.g., 2324 2223).
--output_dir : str
    Directory where the merged CSV files will be saved.

This script will download the corresponding CSV files from football-data.co.uk,
merge them by league, and save the resulting files in the specified output directory.
"""

import os
import argparse
import pandas as pd

# Valid league acronyms
VALID_LEAGUES = ["E0", "E1", "E2", "E3", "EC", "I1", "I2", "D1", "D2", "SP1", "SP2", "F1", "F2"]

# Valid season codes (limiting to recent years only)
VALID_SEASONS = ["2324", "2223", "2122", "2021"]

def validate_leagues(leagues):
    """
    Validates the list of league acronyms.

    Parameters
    ----------
    leagues : list of str
        List of league acronyms to validate.
    
    Raises
    ------
    ValueError
        If any of the league acronyms are not valid.
    """
    for league in leagues:
        if league not in VALID_LEAGUES:
            raise ValueError(f"Invalid league acronym: {league}. Allowed values are {', '.join(VALID_LEAGUES)}")

def validate_seasons(seasons):
    """
    Validates the list of season codes.

    Parameters
    ----------
    seasons : list of str
        List of season codes to validate.
    
    Raises
    ------
    ValueError
        If any of the season codes are not valid.
    """
    for season in seasons:
        if season not in VALID_SEASONS:
            raise ValueError(f"Invalid season code: {season}. Allowed values are {', '.join(VALID_SEASONS)}")

def download_and_merge_data(leagues, seasons, output_dir):
    """
    Downloads and merges football match data from the specified leagues and seasons.

    Parameters
    ----------
    leagues : list of str
        List of league acronyms (e.g., ["E0", "I1", "SP1"]).
    seasons : list of str
        List of season codes (e.g., ["2324", "2223"]).
    output_dir : str
        Directory where the merged CSV files will be saved.
    
    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for league in leagues:
        league_dfs = []
        
        for season in seasons:
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
            try:
                df = pd.read_csv(url)
                league_dfs.append(df)
                print(f"Downloaded data from {url}")
            except Exception as e:
                print(f"Failed to download data from {url}: {e}")
                continue
        
        if league_dfs:
            # Check for consistency in columns
            base_columns = league_dfs[0].columns
            for i, df in enumerate(league_dfs):
                if not base_columns.equals(df.columns):
                    raise ValueError(f"Inconsistent columns in {league} league for season {seasons[i]}")
            
            # Merge dataframes
            merged_df = pd.concat(league_dfs, ignore_index=True)
            
            # Save to CSV
            output_path = os.path.join(output_dir, f"{league}_merged.csv")
            merged_df.to_csv(output_path, index=False)
            print(f"Saved merged data to {output_path}")

def parse_arguments():
    """
    Parses command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Download and merge football data from multiple leagues and seasons.")
    
    parser.add_argument(
        "--leagues", 
        nargs="+", 
        required=True, 
        help="A list of league acronyms (e.g., E0 I1 SP1)."
    )
    
    parser.add_argument(
        "--seasons", 
        nargs="+", 
        required=True, 
        help="A list of season codes (e.g., 2324 2223)."
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory where the merged CSV files will be saved."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate leagues and seasons
    validate_leagues(args.leagues)
    validate_seasons(args.seasons)
    
    # Download and merge data
    download_and_merge_data(
        leagues=args.leagues, 
        seasons=args.seasons,
        output_dir=args.output_dir
    )
