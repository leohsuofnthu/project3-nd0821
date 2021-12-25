from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from starter.starter.ml.model import train_model, compute_model_metrics, inference


def test_process_data(processedTrain):
    """Test function 'process_data'"""

    train = processedTrain["train"]
    test = processedTrain["test"]
    X_train = processedTrain["X_train"]
    y_train = processedTrain["y_train"]
    X_test = processedTrain["X_test"]
    y_test = processedTrain["y_test"]
    encoder = processedTrain["encoder"]
    lb = processedTrain["lb"]

    # check if the output has same length as input
    assert len(train) == len(X_train)
    assert len(train) == len(y_train)
    assert len(test) == len(X_test)
    assert len(test) == len(y_test)

    # check the output encoder and labeller has right type
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)


def test_train_model(processedTrain):
    """Test function 'train_model'"""

    X_train = processedTrain["X_train"]
    y_train = processedTrain["y_train"]
    trained_model = train_model(X_train, y_train)

    # check if the output is the right type of classifier
    assert isinstance(trained_model, RandomForestClassifier)


def test_inference(processedTrain, trainedModel):
    """Test function 'inference'"""

    X_test = processedTrain["X_test"]
    preds = inference(trainedModel, X_test)

    # Check if the prediction has same length as input
    assert len(X_test) == len(preds)


def test_compute_model_metrics(processedTrain, trainedModel):
    """Test function 'compute_model_metrics'"""

    X_test = processedTrain["X_test"]
    y_test = processedTrain["y_test"]
    preds = inference(trainedModel, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # Check if the metric are within proper range
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
