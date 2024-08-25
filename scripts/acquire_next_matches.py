"""
Script to acquire the next matches data from the football-data.org API and update the team names using a mapping file.

Usage:
------
Run this script from the terminal in the root folder as follows:

    python scripts/acquire_next_matches.py --get_teams_names_dir data/processed --next_matches_output_file data/next_matches.json

Parameters:
-----------
--get_teams_names_dir : str
    The directory containing the processed data files, used to extract unique team names.
--next_matches_output_file : str
    The output JSON file to save the updated next matches.

This script will fetch the next matches data from the football-data.org API, read the unique team names from the processed data files,
update the team names in the next matches data using the mapping file, and save the updated next matches to a JSON file.
"""

from dotenv import load_dotenv
import requests
import pandas as pd
from datetime import datetime
import os
import json
import argparse

# Load environment variables from a .env file located in the home directory
load_dotenv(dotenv_path=os.path.expanduser("~/.env"))

# Parameters
API_KEY = os.getenv("API_FOOTBALL_DATA")
BASE_URL = 'https://api.football-data.org/v4'
HEADERS = { 'X-Auth-Token': API_KEY }
COLUMN_NAME = "HomeTeam"  # The column name in the CSV files containing team names


# Dictionary of major competition IDs (expand this list as needed)
COMPETITIONS = {
    'E0': 2021,
    'SP1': 2014,
    'I1': 2019,
    'D1': 2002,
    'F1': 2015,
}

TEAMS_NAMES_MAPPING = {
    'Arsenal FC': 'Arsenal',
    'Brighton & Hove Albion FC': 'Brighton',
    'Brentford FC': 'Brentford',
    'Southampton FC': 'Southampton',
    'Everton FC': 'Everton',
    'AFC Bournemouth': 'Bournemouth',
    'Ipswich Town FC': 'Ipswich',
    'Fulham FC': 'Fulham',
    'Leicester City FC': 'Leicester',
    'Aston Villa FC': 'Aston Villa',
    'Nottingham Forest FC': "Nott'm Forest",
    'Wolverhampton Wanderers FC': 'Wolves',
    'West Ham United FC': 'West Ham',
    'Manchester City FC': 'Man City',
    'Chelsea FC': 'Chelsea',
    'Crystal Palace FC': 'Crystal Palace',
    'Newcastle United FC': 'Newcastle',
    'Tottenham Hotspur FC': 'Tottenham',
    'Manchester United FC': 'Man United',
    'Liverpool FC': 'Liverpool',
    'Villarreal CF': 'Villarreal',
    'RC Celta de Vigo': 'Celta',
    'RCD Mallorca': 'Mallorca',
    'Sevilla FC': 'Sevilla',
    'Rayo Vallecano de Madrid': 'Vallecano',
    'FC Barcelona': 'Barcelona',
    'Real Betis Balompié': 'Betis',
    'Getafe CF': 'Getafe',
    'Athletic Club': 'Ath Bilbao',
    'Valencia CF': 'Valencia',
    'Real Valladolid CF': 'Valladolid',
    'CD Leganés': 'Leganes',
    'Real Sociedad de Fútbol': 'Sociedad',
    'Deportivo Alavés': 'Alaves',
    'Club Atlético de Madrid': 'Ath Madrid',
    'RCD Espanyol de Barcelona': 'Espanol',
    'Girona FC': 'Girona',
    'CA Osasuna': 'Osasuna',
    'UD Las Palmas': 'Las Palmas',
    'Real Madrid CF': 'Real Madrid',
    'Venezia FC': 'Venezia',
    'Torino FC': 'Torino',
    'FC Internazionale Milano': 'Inter',
    'Atalanta BC': 'Atalanta',
    'Bologna FC 1909': 'Bologna',
    'Empoli FC': 'Empoli',
    'US Lecce': 'Lecce',
    'Cagliari Calcio': 'Cagliari',
    'SS Lazio': 'Lazio',
    'AC Milan': 'Milan',
    'SSC Napoli': 'Napoli',
    'Parma Calcio 1913': 'Parma',
    'ACF Fiorentina': 'Fiorentina',
    'AC Monza': 'Monza',
    'Genoa CFC': 'Genoa',
    'Hellas Verona FC': 'Verona',
    'Juventus FC': 'Juventus',
    'AS Roma': 'Roma',
    'Udinese Calcio': 'Udinese',
    'Como 1907': 'Como', 
    '1. FC Union Berlin': 'Union Berlin',
    'FC St. Pauli 1910': None,  # No match found in the list
    'VfB Stuttgart': 'Stuttgart',
    '1. FSV Mainz 05': 'Mainz',
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'TSG 1899 Hoffenheim': 'Hoffenheim',
    'SV Werder Bremen': 'Werder Bremen',
    'Borussia Dortmund': 'Dortmund',
    'VfL Bochum 1848': 'Bochum',
    'Borussia Mönchengladbach': "M'gladbach",
    'Holstein Kiel': None,  # No match found in the list
    'VfL Wolfsburg': 'Wolfsburg',
    'Bayer 04 Leverkusen': 'Leverkusen',
    'RB Leipzig': 'RB Leipzig',
    '1. FC Heidenheim 1846': 'Heidenheim',
    'FC Augsburg': 'Augsburg',
    'FC Bayern München': 'Bayern Munich',
    'SC Freiburg': 'Freiburg',
    'Olympique Lyonnais': 'Lyon',
    'RC Strasbourg Alsace': 'Strasbourg',
    'Stade Brestois 29': 'Brest',
    'AS Saint-Étienne': 'St Etienne',
    'Montpellier HSC': 'Montpellier',
    'FC Nantes': 'Nantes',
    'Toulouse FC': 'Toulouse',
    'Olympique de Marseille': 'Marseille',
    'AS Monaco FC': 'Monaco',
    'Angers SCO': 'Angers',
    'OGC Nice': 'Nice',
    'Le Havre AC': 'Le Havre',
    'AJ Auxerre': 'Auxerre',
    'Stade de Reims': 'Reims',
    'Stade Rennais FC 1901': 'Rennes',
    'Lille OSC': 'Lille',
    'Paris Saint-Germain FC': 'Paris SG',
    'Racing Club de Lens': 'Lens',
    'AC Ajaccio': 'Ajaccio',
    'FC Metz': 'Metz',
}

def get_next_matches(competitions: dict, headers: dict, base_url: str) -> dict:
    """
    Get the next matches for each major league.

    Parameters:
    competitions (dict): Dictionary of competition codes and their corresponding IDs.
    headers (dict): Headers to include in the API request, including the API key.
    base_url (str): Base URL of the football-data.org API.

    Returns:
    dict: Dictionary containing the next matches for each competition.
    """
    next_matches = {}

    for competition, competition_id in competitions.items():
        next_matches[competition] = []

        url = f'{base_url}/competitions/{competition_id}/matches'
        response = requests.get(url, headers=headers)
        data = response.json()

        current_matchday = data['matches'][0]['season']['currentMatchday']  # int
        total_number_of_matches = len(data['matches'])  # int
        last_match_day = data['matches'][-1]['matchday']
        next_matchday = current_matchday + 1 if current_matchday < last_match_day else last_match_day

        print(f'{competition}: Current Matchday {current_matchday}, Total Matches {total_number_of_matches}')  

        for match in data['matches']:
            if match['matchday'] != next_matchday:
                continue

            match_date = datetime.strptime(match['utcDate'], '%Y-%m-%dT%H:%M:%SZ')
            formatted_date = match_date.strftime('%Y-%m-%d %H:%M:%S')
            home_team = match['homeTeam']['name']
            away_team = match['awayTeam']['name']

            print(f'{formatted_date} - {home_team} vs. {away_team}')

            next_matches[competition].append({
                'date': formatted_date,
                'home_team': home_team,
                'away_team': away_team,
            })

    return next_matches       

def read_unique_team_names(directory_path: str, column_name: str) -> list:
    """
    Read all CSV files from the specified directory and extract unique team names.

    Parameters:
    directory_path (str): Path to the directory containing the processed data files.
    column_name (str): The column name in the CSV files that contains the team names.

    Returns:
    list: List of unique team names extracted from the CSV files.
    """
    full_teams_names = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                teams_name = df[column_name].unique().tolist()
                full_teams_names.extend(teams_name)

    return full_teams_names

def replace_team_names(matches_dict: dict, name_mapping: dict) -> dict:
    """
    Replace team names in the next_matches dictionary using the provided name mapping.

    Parameters:
    matches_dict (dict): Dictionary containing the next matches for each competition.
    name_mapping (dict): Dictionary mapping the official team names to their equivalents in full_teams_names.

    Returns:
    dict: Updated matches_dict with team names replaced according to name_mapping.
    """
    for league, matches in matches_dict.items():
        for match in matches:
            if match['home_team'] in name_mapping:
                match['home_team'] = name_mapping[match['home_team']]
            if match['away_team'] in name_mapping:
                match['away_team'] = name_mapping[match['away_team']]
    return matches_dict

def save_to_json(data: dict, filename: str):
    """
    Save the provided dictionary to a JSON file.

    Parameters:
    data (dict): Dictionary to save.
    filename (str): The filename for the JSON file.

    Returns:
    None
    """
    with open(filename, 'w', encoding='utf-16') as json_file:
        json.dump(data, json_file, indent=4)

def parse_arguments():
    """
    Parses command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Acquire next matches data and update team names.")
    
    parser.add_argument(
        "--get_teams_names_dir", 
        type=str,
        required=True, 
        help="The directory containing the processed data files, used to extract unique team names."
    )
    
    parser.add_argument(
        "--next_matches_output_file", 
        type=str,
        required=True, 
        help="The output JSON file to save the updated next matches."
    )
    
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    # Step 1: Fetch the next matches data
    next_matches = get_next_matches(COMPETITIONS, HEADERS, BASE_URL)

    # Step 2: Replace team names in the next matches using the mapping
    next_matches_fd_couk_format = replace_team_names(next_matches, TEAMS_NAMES_MAPPING)

    # Step 3: Save the updated next_matches dictionary to a JSON file
    save_to_json(next_matches_fd_couk_format, args.next_matches_output_file)
