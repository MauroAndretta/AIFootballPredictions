"""
This script contains the tasks to be run by Invoke.

The tasks are:
- data_acquisition: download and merge football match data.
- data_preprocessing: preprocess the raw data.
- train_models: train machine learning models.
- acquire_next_matches: acquire the next football matches data.
- make_predictions: make predictions and generate a Telegram-ready message.
- full_pipeline: run the full pipeline: acquisition, preprocessing, training, predictions

Invoke the full pipeline from the root directory with:

    python -m invoke --search-root scripts full-predictions-pipeline
"""
from invoke import task

@task
def data_acquisition(c, leagues="E0 I1 SP1 F1 D1", seasons="2425 2324 2223", raw_data_output_dir="data/raw"):
    """Task to download and merge football match data."""
    c.run(f"python scripts/data_acquisition.py --leagues {leagues} --seasons {seasons} --raw_data_output_dir {raw_data_output_dir}")

@task
def data_preprocessing(c, raw_data_input_dir="data/raw", processed_data_output_dir="data/processed", num_features=20, clustering_threshold=0.5):
    """Task to preprocess the raw data."""
    c.run(f"python scripts/data_preprocessing.py --raw_data_input_dir {raw_data_input_dir} --processed_data_output_dir {processed_data_output_dir} --num_features {num_features} --clustering_threshold {clustering_threshold}")

@task
def train_models(c, processed_data_input_dir="data/processed", trained_models_output_dir="models", metric_choice="accuracy", n_splits=10, voting="soft"):
    """Task to train machine learning models."""
    c.run(f"python scripts/train_models.py --processed_data_input_dir {processed_data_input_dir} --trained_models_output_dir {trained_models_output_dir} --metric_choice {metric_choice} --n_splits {n_splits} --voting {voting}")

@task
def acquire_next_matches(c, get_teams_names_dir="data/processed", next_matches_output_file="data/next_matches.json"):
    """Task to acquire the next football matches data."""
    c.run(f"python scripts/acquire_next_matches.py --get_teams_names_dir {get_teams_names_dir} --next_matches_output_file {next_matches_output_file}")

@task
def make_predictions(c, models_dir="models", data_dir="data/processed", output_file="telegram_post.txt", json_competitions="data/next_matches.json"):
    """Task to make predictions and generate a Telegram-ready message."""
    c.run(f"python scripts/make_predictions.py --models_dir {models_dir} --data_dir {data_dir} --output_file {output_file} --json_competitions {json_competitions}")

@task
def full_predictions_pipeline(c):
    """Run the full pipeline: acquisition, preprocessing, training, predictions, and README update."""
    data_acquisition(c)
    data_preprocessing(c)
    train_models(c)
    acquire_next_matches(c)
    make_predictions(c)
