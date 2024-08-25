# AIFootballPredictions

**AIFootballPredictions** is a machine learning-based system designed to predict whether a football match will have over 2.5 goals. Leveraging historical data from top European leagues (Serie A, EPL, Bundesliga, La Liga, Ligue 1), it utilizes advanced feature engineering and model training techniques to deliver accurate predictions, making it a valuable tool for sports analytics enthusiasts.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Data Acquisition](#data-acquisition)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
7. [Contributing](#contributing)
8. [License](#license)
9. [Discalimer](#disclaimer)

## Project Overview

AIFootballPredictions aims to create a predictive model to forecast whether a football match will exceed 2.5 goals. The project is divided into three main stages:

1. **Data Acquisition**: Download and merge historical football match data from multiple European leagues.
2. **Data Preprocessing**: Process the raw data to engineer features, handle missing values, and select the most relevant features.
3. **Model Training**: Train several machine learning models, perform hyperparameter tuning, and combine the best models into a voting classifier to make predictions.

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
            └─── `train_models.py`
```

### Key Scripts

- **`data_acquisition.py`**: Downloads and merges football match data from specified leagues and seasons.
- **`data_preprocessing.py`**: Preprocesses the raw data, performs feature engineering, and selects the most relevant features.
- **`train_models.py`**: Trains machine learning models, performs hyperparameter tuning, and saves the best models.

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
This script downloads match data from **football-data.co.uk** for the specified leagues and seasons, merges them, and saves the results to the specified output directory.

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

## Contributing

If you want to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License 

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.

## Disclaimer

This project is intended for educational and informational purposes only. While the AIFootballPredictions system aims to provide accurate predictions for football matches, it is important to understand that predictions are inherently uncertain and should not be used as the sole basis for any decision-making, including betting or financial investments. 

## Disclaimer

This project is intended for educational and informational purposes only. While the AIFootballPredictions system aims to provide accurate predictions for football matches, it is important to understand that predictions are inherently uncertain and should not be used as the sole basis for any decision-making, including betting or financial investments.

The predictions generated by this system can be used as an additional tool during the decision-making process. However, they should be considered alongside other factors and sources of information.

The authors of this project do not guarantee the accuracy, reliability, or completeness of any information provided. Use the predictions at your own risk, and always consider the unpredictability of sports events.

By using this software, you agree that the authors and contributors are not responsible or liable for any losses or damages of any kind incurred as a result of using the software or relying on the predictions made by the system.


The authors of this project do not guarantee the accuracy, reliability, or completeness of any information provided. Use the predictions at your own risk, and always consider the unpredictability of sports events.

By using this software, you agree that the authors and contributors are not responsible or liable for any losses or damages of any kind incurred as a result of using the software or relying on the predictions made by the system.

