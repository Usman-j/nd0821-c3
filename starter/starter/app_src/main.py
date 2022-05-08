#%%
from ast import alias
import os
import joblib
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(0,alias="marital-status")
    occupation: str
    relationship : str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    class Config:
        schema_extra = {
            "description": "Each field is requried.",
            "example": {
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
        }


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

main_dir_path = Path(__file__).parent.parent.parent.absolute()
app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to Census inference API!"}

@app.post("/predict/")
async def predict(data: CensusData):
    print('Data: ', data)
    data = data.dict()
    df_input = pd.DataFrame(data,index=[0])
    print(df_input)
    encoder = joblib.load(os.path.join(main_dir_path,'model','encoder.joblib'))
    rfc_model = joblib.load(os.path.join(main_dir_path,'model','rfc_model.pkl'))
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    X_test, _, _, _ = process_data(
        df_input, cat_features, label=None, training=False, encoder=encoder, lb=None
    )
    y_pred = inference(rfc_model, X_test)
    pred = '>50k' if y_pred == 1 else '<=50k'

    return {"prediction": pred}