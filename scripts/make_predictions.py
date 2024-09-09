"""
AI Football Predictions Script: Predicts the likelihood of over 2.5 goals in upcoming football matches.

This script loads pre-trained machine learning models and match data to predict whether upcoming football matches will end with more than 2.5 goals. The predictions are then formatted into a Telegram-ready message that can be shared directly. 

How to run:
1. Ensure the necessary data and model files are in the specified directories.
2. Run the script with the appropriate arguments to generate predictions.
-----------------------------------------------------------------------------

Example usage, it is suggested to run the script in the root directory:

    python scripts/make_predictions.py --input_leagues_models_dir models --input_data_predict_dir data/processed --final_predictions_out_file data/final_predictions.txt --next_matches data/next_matches.json

Required Libraries:
- pandas
- numpy
- pickle
- argparse
- datetime
- json
"""

import pandas as pd
import os
import json
import pickle
import numpy as np
from datetime import datetime
import argparse

# Define global constants
VALID_LEAGUES = ["E0", "I1", "D1", "SP1", "F1"]

# Define the features for home team, away team, and general match information
HOME_TEAM_FEATURES = [
    'HomeTeam', 'FTHG', 'HG', 'HTHG', 'HS', 'HST', 'HHW', 'HC', 'HF', 'HFKC', 'HO', 'HY', 'HR', 'HBP',
    'B365H', 'BFH', 'BSH', 'BWH', 'GBH', 'IWH', 'LBH', 'PSH', 'SOH', 'SBH', 'SJH', 'SYH', 'VCH', 'WHH',
    'BbMxH', 'BbAvH', 'MaxH', 'AvgH', 'BFEH', 'BbMxAHH', 'BbAvAHH', 'GBAHH', 'LBAHH', 'B365AHH', 'PAHH',
    'MaxAHH', 'AvgAHH', 'BbAHh', 'AHh', 'GBAH', 'LBAH', 'B365AH', 'AvgHomeGoalsScored', 'AvgHomeGoalsConceded',
    'HomeOver2.5Perc', 'AvgLast5HomeGoalsScored', 'AvgLast5HomeGoalsConceded', 'Last5HomeOver2.5Count', 'Last5HomeOver2.5Perc'
]

AWAY_TEAM_FEATURES = [
    'AwayTeam', 'FTAG', 'AG', 'HTAG', 'AS', 'AST', 'AHW', 'AC', 'AF', 'AFKC', 'AO', 'AY', 'AR', 'ABP',
    'B365A', 'BFA', 'BSA', 'BWA', 'GBA', 'IWA', 'LBA', 'PSA', 'SOA', 'SBA', 'SJA', 'SYA', 'VCA', 'WHA',
    'BbMxA', 'BbAvA', 'MaxA', 'AvgA', 'BFEA', 'BbMxAHA', 'BbAvAHA', 'GBAHA', 'LBAHA', 'B365AHA', 'PAHA',
    'MaxAHA', 'AvgAHA', 'AvgAwayGoalsScored', 'AvgAwayGoalsConceded', 'AwayOver2.5Perc', 'AvgLast5AwayGoalsScored',
    'AvgLast5AwayGoalsConceded', 'Last5AwayOver2.5Count', 'Last5AwayOver2.5Perc'
]

"""
The general features are common to both home and away teams and contain match information that is not specific to either team.
This list in no longer necessary because in case that a feature is not in the home or away team features, it will be considered as a general feature.
GENERAL_FEATURES = [
    'Div', 'Date', 'Time', 'FTR', 'Res', 'HTR', 'Attendance', 'Referee', 'Bb1X2', 'BbMxD', 'BbAvD', 'MaxD', 'AvgD',
    'B365D', 'BFD', 'BSD', 'BWD', 'GBD', 'IWD', 'LBD', 'PSD', 'SOD', 'SBD', 'SJD', 'SYD', 'VCD', 'WHD', 'BbOU',
    'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5', 'GB>2.5', 'GB<2.5', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5',
    'Max>2.5', 'Max<2.5', 'Avg>2.5', 'AvgC>2.5', 'Avg<2.5', 'AvgC<2.5', 'MaxCAHA', 'MaxC>2.5', 'B365C<2.5', 'MaxCA',
    'B365CAHH', 'BbAH', 'Over2.5'
]
"""

def load_model(filepath: str):
    """Loads the machine learning model from a specified pickle file.
    
    Args:
        filepath (str): Path to the pickle file containing the model.
    
    Returns:
        model: The loaded machine learning model.
    """
    try:
        with open(filepath, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def load_league_data(filepath: str) -> pd.DataFrame:
    """Loads the league data from a CSV file using pandas.
    
    Args:
        filepath (str): Path to the CSV file containing league data.
    
    Returns:
        pd.DataFrame: The loaded league data as a DataFrame.
    """
    # check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    else:
        print(f"Loading data from {filepath}...")
    # Load the data from the CSV file
        return pd.read_csv(filepath)


def prepare_row_to_predict(home_team_df: pd.DataFrame, away_team_df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """Prepares a DataFrame row for prediction by averaging relevant team statistics.
    
    Args:
        home_team_df (pd.DataFrame): DataFrame containing the home team's data.
        away_team_df (pd.DataFrame): DataFrame containing the away team's data.
        numeric_columns (list): List of numeric columns for prediction.
    
    Returns:
        pd.DataFrame: A single row DataFrame ready for prediction.
    """
    row_to_predict = pd.DataFrame(columns=numeric_columns)
    row_to_predict.loc[len(row_to_predict)] = [None] * len(row_to_predict.columns)

    home_team_final_df = home_team_df.head(5)[numeric_columns]
    away_team_final_df = away_team_df.head(5)[numeric_columns]

    for column in row_to_predict.columns:
        if column in HOME_TEAM_FEATURES:
            row_to_predict.loc[len(row_to_predict)-1, column] = home_team_final_df[column].mean()
        elif column in AWAY_TEAM_FEATURES:
            row_to_predict.loc[len(row_to_predict)-1, column] = away_team_final_df[column].mean()
        # If the column is not in the home or away team features, we take the average of both teams
        else:
            row_to_predict.loc[len(row_to_predict)-1, column] = (away_team_final_df[column].mean() + home_team_final_df[column].mean()) / 2

    return row_to_predict


def make_predictions(league: str, league_model, league_data: pd.DataFrame, competitions: dict) -> str:
    """Makes predictions for a specific league and formats them into a Telegram message.
    
    Args:
        league (str): The league identifier.
        league_model: The machine learning model for the league.
        league_data (pd.DataFrame): DataFrame containing the league data.
        competitions (dict): Dictionary containing competition details and upcoming matches.
    
    Returns:
        str: A formatted string containing the predictions for the league.
    """
    league_section = ""
    for competition_league, competitions_info in competitions.items():
        if competition_league == league:
            league_section = f"**{competitions_info['name']}**:\n"
            for match in competitions_info["next_matches"]:
                home_team = match['home_team']
                away_team = match['away_team']

                if home_team not in league_data['HomeTeam'].values or away_team not in league_data['AwayTeam'].values:
                    continue

                home_team_df = league_data[league_data['HomeTeam'] == home_team]
                away_team_df = league_data[league_data['AwayTeam'] == away_team]

                numeric_columns = league_data.select_dtypes(include=['number']).columns
                if 'Over2.5' in numeric_columns:
                    numeric_columns = numeric_columns.drop('Over2.5')

                row_to_predict = prepare_row_to_predict(home_team_df, away_team_df, numeric_columns)
                X_test = row_to_predict.values
                prediction = league_model.predict(X_test)
                predicted_probability = league_model.predict_proba(X_test)[0]

                if prediction == 1:
                    result = f"Over 2.5 Goals! 🔥 ({round(predicted_probability[1] * 100, 2)}% chance)"
                else:
                    result = f"Under 2.5 Goals ({round(predicted_probability[0] * 100, 2)}% chance)"

                league_section += f"- ⚽ **{home_team}** 🆚 **{away_team}**: {result}\n"

    return league_section


def main(input_leagues_models_dir: str, input_data_predict_dir: str, final_predictions_out_file: str, next_matches: str):
    """Main function that handles the entire prediction process.
    
    Args:
        input_leagues_models_dir (str): Directory containing the model files.
        input_data_predict_dir (str): Directory containing the processed data files.
        final_predictions_out_file (str): Path where the output Telegram message will be saved.
        next_matches (str): Path to the JSON file with upcoming matches information.
    """
    try:
        print("Loading JSON file with upcoming matches...\n")
        with open(next_matches, 'r', encoding='utf-16') as json_file:
            competitions = json.load(json_file)
    except Exception as e:
        raise Exception(f"Error loading JSON file: {e}")

    predictions_message = f"🎯 **AI Football Predictions: Will There Be Over 2.5 Goals?** 🎯\n\nCheck out the latest predictions for the upcoming football matches! We've analyzed the data and here are our thoughts:\n PREDICTIONS DONE: {datetime.now().strftime('%Y-%m-%d')} \n\n"

    for league in VALID_LEAGUES:
        print(f"----------------------------------")
        print(f"\nMaking predictions for {league}...\n")
        model_path = os.path.join(input_leagues_models_dir, f"{league}_voting_classifier.pkl")
        data_path = os.path.join(input_data_predict_dir, f"{league}_merged_preprocessed.csv")

        if not os.path.exists(model_path) or not os.path.exists(data_path):
            print(f"Missing data or model for {league}. Skipping...")
            continue

        league_model = load_model(model_path)
        league_data = load_league_data(data_path)
        print(f"Loaded model and data for {league}.")
        print(f"Predicting matches for {league}...")
        league_section = make_predictions(league, league_model, league_data, competitions)
        print(f"Predictions made for {league}.")
        predictions_message += league_section + "\n"

    with open(final_predictions_out_file, 'w', encoding='utf-8') as file:
        file.write(predictions_message)
        print(f"\n Predictions saved to {final_predictions_out_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Football Predictions Script")
    parser.add_argument('--input_leagues_models_dir', type=str, required=True, help="Directory containing the model files")
    parser.add_argument('--input_data_predict_dir', type=str, required=True, help="Directory containing the processed data files")
    parser.add_argument('--final_predictions_out_file', type=str, required=True, help="File path to save the Telegram message output")
    parser.add_argument('--next_matches', type=str, required=True, help="Path to the JSON file with upcoming matches information")

    args = parser.parse_args()
    main(args.input_leagues_models_dir, args.input_data_predict_dir, args.final_predictions_out_file, args.next_matches)
