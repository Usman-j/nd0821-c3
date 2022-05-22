# Script to train machine learning model.
#%%
# Add the necessary imports for the starter code.
import os
from pathlib import Path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics, validate_model_per_slice
#%%
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
main_dir_path = Path(__file__).parent.parent.absolute()
# Add code to load in the data.
df_census = pd.read_csv(os.path.join(main_dir_path,'data','census_cleaned.csv'))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df_census, test_size=0.20, stratify=df_census['salary'], random_state=42)

#%%
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Saving encoder and lb
joblib.dump(encoder, os.path.join(main_dir_path,'model','encoder.joblib'))
joblib.dump(lb, os.path.join(main_dir_path,'model','binarizer.joblib'))
# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
trained_model = train_model(X_train, y_train)
joblib.dump(trained_model, os.path.join(main_dir_path,'model','rfc_model.pkl'))
#%% Model Inference
rfc_model = joblib.load(os.path.join(main_dir_path,'model','rfc_model.pkl'))
y_pred = inference(rfc_model, X_test)
test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, y_pred)
logger.info(f"Test data metrics; Precision: {test_precision}, \
    Recall: {test_recall}, fbeta: {test_fbeta}") 

#%% Compute model metrics on slices of data
encoder = joblib.load(os.path.join(main_dir_path,'model','encoder.joblib'))
lb = joblib.load(os.path.join(main_dir_path,'model','binarizer.joblib'))
validate_model_per_slice(rfc_model, test, cat_features, encoder, lb, os.path.join(main_dir_path,'model','slice_output.txt'))