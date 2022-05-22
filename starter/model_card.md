# Model Card for Census Income dataset

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Random Forest classifier was trained by using 5-fold GridSearchCV on 'n_estimators' and 'max_depth' with optimal values found to be at 500 and 20 respecively. 
## Intended Use
This model is intended to predict whether annual salary of a person is above or below $50k based on demographics and financial data.
## Training Data
The data was obtained from [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income). Original data had 32561 samples which reduced to 30139 after dropping duplicates and missing values after the EDA process. 80% of the data is used for training and stratified with respect to the target variable of 'salary'. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.
## Evaluation Data
20% of the data used for evaluation.
## Metrics
The evaluation metrics used were Precision, Recall and fbeta (beta=1) with values 0.78, 0.62 and 0.69 respectively.

## Ethical Considerations
Model was evaluated on data slices as per the attributes of race, gender and nationality to investigate any significant bias present in the model performance.
## Caveats and Recommendations
Performance metrics on various data slices should be carefully considered in accordance with the target population upon deployment and any signficant data imbalance should be mitigated if necessary.
