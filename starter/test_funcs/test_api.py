import json
from fastapi.testclient import TestClient
from starter.app_src.main import app

client = TestClient(app)

def test_api_locally_get_root():
    '''
    Test GET method at root of app
    '''
    r = client.get("/")
    assert r.status_code == 200
    assert r.request.method == "GET"
    assert r.json()['greeting'] == "Welcome to Census inference API!"

def test_api_predict_class0():
    '''
    Test POST method at /predict/ of app for a class 0 sample
    '''
    data = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        }
    r = client.post("/predict/", json=data)
    assert r.status_code == 200
    assert r.json()['prediction'] == '<=50k'

def test_api_predict_class1():
    '''
    Test POST method at /predict/ of app for a class 1 sample
    '''
    data = {
            "age": 56,
            "workclass": "Local-gov",
            "fnlgt": 216851,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Tech-support",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        }
    r = client.post("/predict/", json=data)
    assert r.status_code == 200
    assert r.json()['prediction'] == '>50k'

def test_api_predict_missing_feats():
    '''
    Test POST method at /predict/ of app for sample with missing feature(s)
    '''
    data = {
            "age": 56,
            "workclass": "Local-gov",
            "fnlgt": 216851,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Tech-support",
            "relationship": "Husband",
        }
    r = client.post("/predict/", json=data)
    assert r.status_code == 422
    assert r.reason == 'Unprocessable Entity'