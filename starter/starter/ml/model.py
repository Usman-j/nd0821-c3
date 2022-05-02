import logging
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from starter.ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto'],
        'max_depth': [4, 20, 100],
        'criterion': ['gini']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    return cv_rfc.best_estimator_

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def validate_model_per_slice(trained_model, df_test, cat_feats,
                            encoder, lb, output_path):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    trained_model : RandomForestClassifier
        Trained machine learning model.
    df_test : pd.DataFrame
        Test Data frame for validation.
    cat_feats : list[str]
        List containing the names of the categorical features.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    output_path : str
        Path of text file for writing the output of validation results.
    Returns
    -------
    
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()
    with open(output_path,'w') as output_file:
        for cat in cat_feats:
            for cat_type in df_test[cat].unique():
                df_cat_type = df_test[df_test[cat]==cat_type]

                X_test, y_test, _, _ = process_data(
                    df_cat_type, cat_feats, label="salary", training=False, encoder=encoder, lb=lb
                )
                y_pred = trained_model.predict(X_test)
                test_precision, test_recall, test_fbeta = compute_model_metrics(y_test, y_pred)
                slice_metrics = f"{cat}_{cat_type}; Precision: {test_precision}, Recall: {test_recall}, fbeta: {test_fbeta}"
                logger.info(slice_metrics)
                output_file.write(slice_metrics + '\n')
  