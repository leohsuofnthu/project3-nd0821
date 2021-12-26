# API test locally


def test_api_locally_welcome(client):
    """Local test for GET method 'welcome'"""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"greeting": "Welcome to my model !"}


def test_api_locally_model_inference_0(client):
    """Local test for POST method 'model_inference' with output <=50K"""
    sample = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_api_locally_model_inference_0(client):
    """Local test for POST method 'model_inference' with output >50K"""
    sample = {
        "age": 49,
        "workclass": "Self-emp-inc",
        "fnlgt": 191681,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States",
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
