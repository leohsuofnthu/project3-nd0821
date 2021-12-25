from ml.data import process_data
from ml.model import inference, compute_model_metrics


def data_slicing_categorical(test, cat_features, trained_model, encoder, lb, col):
    """Function for data slicing model performance given certain categorical column"""

    # get distinct column category value
    unique_val = test[col].unique()
    metrics = {}

    # iterate each value and record the metrics
    for val in unique_val:
        # Fix the feature
        idx = test[col] == val
        temp_test = test[idx]

        # Process this subset of data for testing
        X_test, y_test, encoder, lb = process_data(
            temp_test,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        # Do the inference and Compute the metrics
        preds = inference(trained_model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        metrics[val] = {"precision": precision, "recall": recall, "fbeta": fbeta}

    return metrics
