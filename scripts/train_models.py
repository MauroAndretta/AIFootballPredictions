"""
Script to train machine learning models to predict the Over2.5 outcome of football matches.

Usage:
------
Run this script from the terminal in the root directory as follows:

    python scripts/train_models.py --processed_data_input_dir data/processed --trained_models_output_dir models

Parameters:
-----------
--processed_data_input_dir : str
    Path to the folder containing the CSV files to be used for training (e.g., 'data/preprocessed/').
--trained_models_output_dir : str
    Path to the folder where the trained models will be saved (e.g., 'models/').
--metric_choice : str
    The metric to use for hyperparameter tuning. Choose from 'accuracy', 'precision', 'f1', or 'roc_auc'.
--n_splits : int
    Number of splits for cross-validation.
--voting : str
    Voting method for the ensemble model. Choose from 'soft' or 'hard'.

The script processes each CSV file individually, trains several machine learning models, performs hyperparameter
tuning, combines the best models into a voting classifier, and saves the trained voting classifier for each league.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Suppress the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    --------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train models to predict Over2.5 football outcomes.")
    parser.add_argument('--processed_data_input_dir', type=str, required=True, help="Path to the folder containing CSV files.")
    parser.add_argument('--trained_models_output_dir', type=str, required=True, help="Path to the folder to save trained models.")
    # Specify the allowed choices for metric_choice
    parser.add_argument('--metric_choice', type=str, choices=['accuracy', 'precision', 'f1', 'roc_auc'], default='accuracy',
                        help="The metric to use for hyperparameter tuning. Choose from 'accuracy', 'precision', 'f1', or 'roc_auc'.")
    parser.add_argument('--n_splits', type=int, default=10, help="Number of splits for cross-validation.")
    parser.add_argument('--voting', type=str, choices=['soft', 'hard'], default='soft', help="Voting method for the ensemble model.")
    return parser.parse_args()


def load_data(processed_data_input_dir: str) -> dict:
    """
    Load CSV files from the specified folder and return a dictionary of DataFrames.

    Parameters:
    -----------
    processed_data_input_dir : str
        The path to the folder containing the CSV files.

    Returns:
    --------
    data : dict
        A dictionary where keys are file names (without extension) and values are DataFrames.
    """
    data = {}
    for file_name in os.listdir(processed_data_input_dir):
        if file_name.endswith('.csv'):
            league_name = file_name.split('_')[0]
            file_path = os.path.join(processed_data_input_dir, file_name)
            data[league_name] = pd.read_csv(file_path)
    return data


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare the feature matrix X and the target variable y from the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the preprocessed data.

    Returns:
    --------
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The target variable.
    """
    y = df['Over2.5'].values
    numerical_columns = df.select_dtypes(include=['number']).columns
    X = df[numerical_columns].drop(columns=['Over2.5']).values
    return X, y


def train_and_save_models(X: np.ndarray, y: np.ndarray, trained_models_output_dir: str, league_name: str, metric_choice: str, voting: str = 'soft', n_splits: int = 10):
    """
    Train models, perform hyperparameter tuning, create a voting classifier, and save the model.

    Parameters:
    -----------
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The target variable.
    trained_models_output_dir : str
        The folder where the trained models will be saved.
    league_name : str
        The name of the league, used for naming the saved model file.
    metric_choice : str
        The metric to use for hyperparameter tuning.
    n_splits : int
        Number of splits for cross-validation
    
    """
    # Define models and hyperparameters
    lr_model = LogisticRegression(random_state=42)
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],  # Include only solvers that support 'l1' and 'l2'
        'max_iter': [2000, 3000]
    }

    knn_model = KNeighborsClassifier()
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    svm_model = SVC(probability=True)
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4, 5]
    }

    rf_model = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'bootstrap': [True]
    }

    xgb_model = XGBClassifier(tree_method="hist", eval_metric='logloss')
    xgb_param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    hgb_model = HistGradientBoostingClassifier(random_state=42)
    hgb_param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_iter': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'l2_regularization': [0.0, 0.1, 0.5],
        'early_stopping': [True]
    }

    # Define scoring metrics
    # Assume the user input is stored in the variable `metric_choice`
    if metric_choice == 'accuracy':
        scorer = make_scorer(accuracy_score)
    elif metric_choice == 'precision':
        scorer = make_scorer(precision_score)
    elif metric_choice == 'f1':
        scorer = make_scorer(f1_score)
    elif metric_choice == 'roc_auc':
        scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
    else:
        print("Invalid metric choice. Please select from 'accuracy', 'precision', 'f1', or 'roc_auc'.")
        scorer = make_scorer(accuracy_score)


    # 10-fold cross-validation
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Combine the models and hyperparameters into a dictionary
    models = {
        'Logistic Regression': (lr_model, lr_param_grid),
        'KNN': (knn_model, knn_param_grid),
        'SVM': (svm_model, svm_param_grid),
        'Random Forest': (rf_model, rf_param_grid),
        'XGBoost': (xgb_model, xgb_param_grid),
        'HistGradientBoosting': (hgb_model, hgb_param_grid),
    }

    results = {}
    best_params = {}

    for model_name, (model, param_grid) in models.items():
        print(f"Evaluating {model_name}...")

        # Initialize HalvingGridSearchCV with the inner cross-validation and hyperparameter grid
        grid_search = HalvingGridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scorer, verbose=0)

        # Fit the grid search on the whole dataset to get the best parameters
        grid_search.fit(X, y)

        # Get cross-validated score
        cv_score = cross_val_score(grid_search.best_estimator_, X, y, cv=cv, scoring=scorer)

        # Store the results and best parameters
        results[model_name] = cv_score
        best_params[model_name] = grid_search.best_params_

        print(f"{model_name} - {scorer._score_func.__name__}: {np.mean(cv_score):.4f} ± {np.std(cv_score):.4f}")
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    # Initialize the models with the best hyperparameters
    best_lr_model = LogisticRegression(**best_params['Logistic Regression'])
    best_knn_model = KNeighborsClassifier(**best_params['KNN'])
    best_svm_model = SVC(**best_params['SVM'], probability=True)
    best_rf_model = RandomForestClassifier(**best_params['Random Forest'])
    best_xgb_model = XGBClassifier(**best_params['XGBoost'])
    best_hgb_model = HistGradientBoostingClassifier(**best_params['HistGradientBoosting'])

    print("Training Voting Classifier, an ensamble of the best models...")

    # Combine the models into a voting classifier
    voting_clf = VotingClassifier(estimators=[
        ('lr', best_lr_model),
        ('knn', best_knn_model),
        ('svm', best_svm_model),
        ('rf', best_rf_model),
        ('xgb', best_xgb_model),
        ('hgb', best_hgb_model)
    ], voting=voting)  # 'soft' for probability-based voting, 'hard' for majority voting

    # Fit the voting classifier
    voting_clf.fit(X, y)

    # Evaluate the ensemble using cross-validation
    cv_scores = cross_val_score(voting_clf, X, y, cv=cv, scoring=scorer)
    print(f"Voting Classifier - {scorer._score_func.__name__}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Save the model
    model_filename = os.path.join(trained_models_output_dir, f"{league_name}_voting_classifier.pkl")
    with open(model_filename, 'wb') as f:
        pickle.dump(voting_clf, f)
    print(f"Model saved to {model_filename}")


def main():

    try:
        # Parse arguments
        args = parse_arguments()

        # Load data
        data = load_data(args.processed_data_input_dir)

        # Ensure output directory exists
        os.makedirs(args.trained_models_output_dir, exist_ok=True)

        # Train and save models for each league
        for league_name, df in data.items():
            print(f"Processing league: {league_name}")
            X, y = prepare_data(df)
            train_and_save_models(X, y, args.trained_models_output_dir, league_name, args.metric_choice, args.voting, args.n_splits)
        
    except Exception as e:
        raise (f"An error occurred: {e}")


if __name__ == "__main__":
    main()
