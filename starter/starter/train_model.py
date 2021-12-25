# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import joblib
import pandas as pd
from ml.data import process_data
from ml.model import train_model

# Add code to load in the data.
dirname = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(dirname, "../data/cleaned_census.csv"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False
)
# Train and save a model.
trained_model = train_model(X_train, y_train)
joblib.dump(train_model, os.path.join(dirname, "../model/model.joblib"))
