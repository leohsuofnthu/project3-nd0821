# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import joblib
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model
from data_slice import data_slicing_categorical

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
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
# Train and save model and encoder
trained_model = train_model(X_train, y_train)
joblib.dump(trained_model, os.path.join(dirname, "../model/model.joblib"))
joblib.dump(encoder, os.path.join(dirname, "../model/encoder.joblib"))

# Data slicing function on certain column (using education as example)
# using list as input, so as to specify multiple columns with flexibilitus
data_slicing_categorical(test, cat_features, trained_model, encoder, lb, "education")

# Show the overall performance for writing model card
preds = inference(trained_model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"Overall Performance: precision:{precision}, recall:{recall}, fbeta:{fbeta}")
