"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

import joblib

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    np.random.seed(12345)
    
    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    
    # model
    # {'learning_rate': 0.1, 'max_depth': 20, 'n_estimators': 100, 'num_leaves': 31}

    model = lgb.LGBMClassifier(random_state=12345, learning_rate=0.1, max_depth=20, n_estimators=100, num_leaves=31)

    # Fit the model
    model.fit(model_df.drop(['new_child','nomem_encr'], axis=1),
               model_df['new_child'])

    # Save the model
    joblib.dump(model, "model.joblib")


# Load the cleaned data and outcome data
cleaned_df = pd.read_csv(r"C:\Users\guerraur\OneDrive - Tilburg University\Data_challenge\cleaned_data\cleaned_lGBM.csv")
outcome_df = pd.read_csv(r"C:\Users\guerraur\OneDrive - Tilburg University\Data_challenge\cleaned_data\outcome.csv")

# Train and save the model
train_save_model(cleaned_df, outcome_df)
