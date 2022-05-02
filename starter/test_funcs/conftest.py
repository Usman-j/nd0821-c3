#%%
import pytest
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


@pytest.fixture
def train_data():
    # main_dir_path = Path(__file__).parent.parent.absolute()
    df_census = pd.read_csv('starter/data/census_cleaned.csv')
    train, _ = train_test_split(df_census, test_size=0.20, stratify=df_census['salary'], random_state=42)
    return train

@pytest.fixture
def categorical_feats():
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return cat_features