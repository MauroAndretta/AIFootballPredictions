# AIFootballPredictions

🎯 **AI Football Predictions: Will There Be Over 2.5 Goals?** 🎯

Check out the latest predictions for the upcoming football matches! We've analyzed the data and here are our thoughts:
 PREDICTIONS DONE: 2025-02-28 

**Premier League**:

**Serie A**:
- ⚽ **Fiorentina** 🆚 **Lecce**: Under 2.5 Goals (59.89% chance)
- ⚽ **Atalanta** 🆚 **Venezia**: Over 2.5 Goals! 🔥 (85.17% chance)
- ⚽ **Napoli** 🆚 **Inter**: Over 2.5 Goals! 🔥 (94.92% chance)
- ⚽ **Udinese** 🆚 **Parma**: Over 2.5 Goals! 🔥 (75.26% chance)
- ⚽ **Monza** 🆚 **Torino**: Over 2.5 Goals! 🔥 (77.75% chance)
- ⚽ **Bologna** 🆚 **Cagliari**: Under 2.5 Goals (76.23% chance)
- ⚽ **Genoa** 🆚 **Empoli**: Under 2.5 Goals (78.29% chance)
- ⚽ **Roma** 🆚 **Como**: Over 2.5 Goals! 🔥 (50.44% chance)
- ⚽ **Milan** 🆚 **Lazio**: Under 2.5 Goals (54.49% chance)
- ⚽ **Juventus** 🆚 **Verona**: Under 2.5 Goals (88.96% chance)

**Bundesliga**:
- ⚽ **Stuttgart** 🆚 **Bayern Munich**: Over 2.5 Goals! 🔥 (90.58% chance)
- ⚽ **St Pauli** 🆚 **Dortmund**: Under 2.5 Goals (78.8% chance)
- ⚽ **Bochum** 🆚 **Hoffenheim**: Over 2.5 Goals! 🔥 (76.12% chance)
- ⚽ **Werder Bremen** 🆚 **Wolfsburg**: Under 2.5 Goals (81.46% chance)
- ⚽ **RB Leipzig** 🆚 **Mainz**: Over 2.5 Goals! 🔥 (87.95% chance)
- ⚽ **Heidenheim** 🆚 **M'gladbach**: Over 2.5 Goals! 🔥 (68.89% chance)
- ⚽ **Ein Frankfurt** 🆚 **Leverkusen**: Under 2.5 Goals (61.12% chance)
- ⚽ **Union Berlin** 🆚 **Holstein Kiel**: Over 2.5 Goals! 🔥 (91.03% chance)
- ⚽ **Augsburg** 🆚 **Freiburg**: Over 2.5 Goals! 🔥 (57.77% chance)

**La Liga**:
- ⚽ **Valladolid** 🆚 **Las Palmas**: Under 2.5 Goals (94.32% chance)
- ⚽ **Girona** 🆚 **Celta**: Over 2.5 Goals! 🔥 (73.47% chance)
- ⚽ **Vallecano** 🆚 **Sevilla**: Over 2.5 Goals! 🔥 (88.3% chance)
- ⚽ **Betis** 🆚 **Real Madrid**: Over 2.5 Goals! 🔥 (94.72% chance)
- ⚽ **Ath Madrid** 🆚 **Ath Bilbao**: Over 2.5 Goals! 🔥 (94.72% chance)
- ⚽ **Leganes** 🆚 **Getafe**: Over 2.5 Goals! 🔥 (65.62% chance)
- ⚽ **Barcelona** 🆚 **Sociedad**: Over 2.5 Goals! 🔥 (72.24% chance)
- ⚽ **Mallorca** 🆚 **Alaves**: Under 2.5 Goals (99.17% chance)
- ⚽ **Osasuna** 🆚 **Valencia**: Over 2.5 Goals! 🔥 (63.72% chance)
- ⚽ **Villarreal** 🆚 **Espanol**: Over 2.5 Goals! 🔥 (89.97% chance)

**Ligue 1**:
- ⚽ **Monaco** 🆚 **Reims**: Over 2.5 Goals! 🔥 (73.58% chance)
- ⚽ **St Etienne** 🆚 **Nice**: Under 2.5 Goals (93.4% chance)
- ⚽ **Lens** 🆚 **Le Havre**: Over 2.5 Goals! 🔥 (65.85% chance)
- ⚽ **Paris SG** 🆚 **Lille**: Over 2.5 Goals! 🔥 (66.92% chance)
- ⚽ **Lyon** 🆚 **Brest**: Over 2.5 Goals! 🔥 (92.07% chance)
- ⚽ **Montpellier** 🆚 **Rennes**: Over 2.5 Goals! 🔥 (76.95% chance)
- ⚽ **Auxerre** 🆚 **Strasbourg**: Under 2.5 Goals (70.88% chance)
- ⚽ **Angers** 🆚 **Toulouse**: Over 2.5 Goals! 🔥 (78.66% chance)
- ⚽ **Marseille** 🆚 **Nantes**: Under 2.5 Goals (56.8% chance)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Data Acquisition](#data-acquisition)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [Upcoming Matches Acquisition](#upcoming-matches-acquisition)
    - [Set up the API_KEY](#setu-up-the-api_key)
8. [Making Predictions](#making-predictions)
9. [Supported Leagues](#supported-leagues)
10. [Contributing](#contributing)
11. [License](#license)
12. [Disclaimer](#disclaimer)

## Project Overview

AIFootballPredictions aims to create a predictive model to forecast whether a football match will exceed 2.5 goals. The project is divided into four main stages:

1. **Data Acquisition**: Download and merge historical football match data from multiple European leagues.
2. **Data Preprocessing**: Process the raw data to engineer features, handle missing values, and select the most relevant features.
3. **Model Training**: Train several machine learning models, perform hyperparameter tuning, and combine the best models into a voting classifier to make predictions.
4. **Making Predictions**: Use the trained models to predict outcomes for upcoming matches and generate a formatted message for sharing.

## Directory Structure

The project is organized into the following directories:

```
└─── `AIFootballPredictions`
    ├─── `conda`: all the conda environemnts
    ├─── `data`: the folder for the data
    │       ├─── `processed`
    │       └─── `raw`
    ├─── `models`: the folder with the saved and trained models
    ├─── `notebooks`: all the notebooks if any
    └─── `scripts`: all the python scripts
            ├─── `data_acquisition.py`
            ├─── `data_preprocessing.py`
            ├─── `train_models.py`
            ├─── `acquire_next_matches.py`
            └─── `make_predictions.py`
```


### Key Scripts

- **`data_acquisition.py`**: Downloads and merges football match data from specified leagues and seasons.
- **`data_preprocessing.py`**: Preprocesses the raw data, performs feature engineering, and selects the most relevant features.
- **`train_models.py`**: Trains machine learning models, performs hyperparameter tuning, and saves the best models.
- **`acquire_next_matches.py`**: Acquires the next football matches data, updates team names using a mapping file, and saves the results to a JSON file.
- **`make_predictions.py`**: Uses the trained models to predict outcomes for upcoming matches and formats the results into a readable txt message.

**Note**: it is suggested to avoid path error, to execute all the scripts in the root folder. 

## Setup and Installation

To set up the environment for this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/AIFootballPredictions.git
   cd AIFootballPredictions
   ```

2. Create a conda environment

   ```bash
   conda env create -f conda/aifootball_predictions.yaml
   conda activate aifootball_predictions
   ```

## Data Acquisition

To download and merge football match data, run the `data_acquisition.py` script:

```bash
python scripts/data_acquisition.py --leagues E0 I1 SP1 F1 D1 --seasons 2425 2324 2223 --raw_data_output_dir data/raw
```
This script downloads match data from [football-data.co.uk](https://www.football-data.co.uk/) for the specified leagues and seasons, merges them, and saves the results to the specified output directory.

To avoid error please see the [Supported Leagues](#supported-leagues) sections. 

## Data Preprocessing

Once the raw data is downloaded, preprocess it by running the `data_preprocessing.py` script:

```bash
python scripts/data_preprocessing.py --raw_data_input_dir data/raw --processed_data_output_dir data/processed --num_features 20 --clustering_threshold 0.5
```
This script processes each CSV file in the input folder, performs feature engineering, selects relevant features while addressing feature correlation, handles missing values, and saves the processed data.

## Model Training

To train machine learning models and create a voting classifier, use the `train_models.py` script:

```bash
python scripts/train_models.py --processed_data_input_dir data/processed --trained_models_output_dir models --metric_choice accuracy --n_splits 10 --voting soft
```
This script processes each CSV file individually, trains several machine learning models, performs hyperparameter tuning, combines the best models into a voting classifier, and saves the trained voting classifier for each league.

## Upcoming Matches Acquisition

To acquire the next football matches data and update the team names, run the `acquire_next_matches.py` script:

```bash
python scripts/acquire_next_matches.py --get_teams_names_dir data/processed --next_matches_output_file data/next_matches.json
```
This script will:

- Fetch the next matches data from the [football-data.org API](https://www.football-data.org/).
- Read the unique team names from the processed data files.
- Update the team names in the next matches data using the mapping file.
    - This step is necessary because the teams' names acquired with the [football-data.org API](https://www.football-data.org/) differ from the teams' names acquired from [football-data.co.uk](https://www.football-data.co.uk/), which've been used to train the ML models. 
- Save the updated next matches to a JSON file.

### Setu up the API_KEY 

In order to properly execute the `acquire_next_matches.py` script it is first necessary to set up the API_KEY to gather the next matches information. Below the procedure on how to properly set up the variable:

1. **Register for an API Key:**
   - Go to the [Football-Data.org website](https://www.football-data.org/) and register to get your personal API key.

2. **Create a `~/.env` File:**
   - This file will be used by the `load_dotenv` library to set up the `API_FOOTBALL_DATA` environment variable.
   - To create the file:
     - Open your terminal and run the command: `vim ~/.env`
     - This will create a new `~/.env` file if it doesn't already exist.

3. **Insert the API Key:**
   - After running the `vim` command, press the `i` key (for "insert mode").
   - Write down the following line, replacing `your_personal_key` with your actual API key:
     - `API_FOOTBALL_DATA=your_personal_key`

4. **Save and Exit:**
   - Press the `Esc` key to exit insert mode.
   - Then, type `:wq!` and press `Enter` to save the changes and exit the editor.

5. **Verify the Variable:**
   - To check if the variable has been properly set, run the following command from the terminal:
     - `cat ~/.env`
   - You should see the `API_FOOTBALL_DATA` variable listed with your API key.

## Making Predictions

To predict the outcomes for upcoming matches and generate a formatted message for sharing, run the `make_predictions.py` script:

```bash
python scripts/make_predictions.py --models_dir models --data_dir data/processed --output_file final_predictions.txt --json_competitions data/next_matches.json
```
This script will:

- Load the pre-trained models and the processed data.
- Make predictions for upcoming matches based on the next matches data.
- Format the predictions into a redable `.txt` message and save it to the specified output file.

## Supported Leagues

For the moment, the team name mapping has been done manually. The predictions currently support the following leagues:

- *Premier League*: **E0**
- *Serie A*: **I1**
- *Ligue 1*: **F1**
- *La Liga (Primera Division)*: **SP1**
- *Bundesliga*: **D1**

For this reason be carful when executing the [data acquisition](#data-acquisition) step. 

## Contributing

If you want to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [BSD-3-Claude license](LICENSE) - see the `LICENSE` file for details.

## Disclaimer

This project is intended for educational and informational purposes only. While the AIFootballPredictions system aims to provide accurate predictions for football matches, it is important to understand that predictions are inherently uncertain and should not be used as the sole basis for any decision-making, including betting or financial investments.

The predictions generated by this system can be used as an additional tool during the decision-making process. However, they should be considered alongside other factors and sources of information.

The authors of this project do not guarantee the accuracy, reliability, or completeness of any information provided. Use the predictions at your own risk, and always consider the unpredictability of sports events.

By using this software, you agree that the authors and contributors are not responsible or liable for any losses or damages of any kind incurred as a result of using the software or relying on the predictions made by the system.

