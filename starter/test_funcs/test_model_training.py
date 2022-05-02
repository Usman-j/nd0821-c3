import os
from pathlib import Path
import pytest
import numpy as np
import sklearn
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics

def test_process_training_data(train_data, categorical_feats):
    '''
    Test 'process_data' function for training mode
    '''
    X_train, y_train, encoder, lb = process_data(
        train_data, categorical_features=categorical_feats, label="salary", training=True
    )
    
    assert type(X_train) == np.ndarray
    assert type(y_train) == np.ndarray
    assert type(encoder) == sklearn.preprocessing._encoders.OneHotEncoder
    assert type(lb) == sklearn.preprocessing._label.LabelBinarizer

def test_train_model():
    '''
    Test 'train_model' function
    '''
    X_train = np.random.randint(0,100, (1000,10))
    y_train = np.random.randint(0,2, (1000,))
    trained_model = train_model(X_train, y_train)
    assert type(trained_model) == sklearn.ensemble._forest.RandomForestClassifier

def test_compute_model_metrics():
    '''
    Test 'compute_model_metrics' function
    '''
    y_test = np.random.randint(0,2, (200,))
    y_pred = np.random.randint(0,2, (200,))
    test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, y_pred)
    assert type(test_precision) == np.float64
    assert type(test_recall) == np.float64
    assert type(test_fbeta) == np.float64