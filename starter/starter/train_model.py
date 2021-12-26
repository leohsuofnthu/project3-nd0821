# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import joblib
import pandas as pd
from ml.data import process_data
from ml.model import train_model
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

# Data slicing, performance evaluation
res = {}
for cat in cat_features:
    res[cat] = data_slicing_categorical(
        test, cat_features, trained_model, encoder, lb, cat
    )

# Write the result into slice_output.txt
with open(os.path.join(dirname, "../screenshots/slice_output.txt"), "w") as f:
    for cat in res.keys():
        f.write(f"{cat}\n")
        for value in res[cat].keys():
            f.write(f"\t {value.strip()}\n")
            f.write(
                f"\t\t precision:{res[cat][value]['precision']} recall:{res[cat][value]['precision']} fbeta:{res[cat][value]['precision']}\n"
            )
