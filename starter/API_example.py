#%%
import requests
import json
#%%
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
'''
Local API example
'''
# r = requests.post("http://127.0.0.1:8000/predict/", data= json.dumps(data))
# print(r.json())

'''
Live API example
'''
r = requests.post("https://usman-census-api.herokuapp.com/predict/", data= json.dumps(data))
print(r.json())