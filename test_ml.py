import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
import pytest


@pytest.fixture
def sample_data():
    data = {
        "age": [39, 50, 38, 53, 28],
        "workclass": [
            "State-gov",
            "Self-emp-not-inc",
            "Private",
            "Private",
            "Private"
        ],
        "fnlgt": [77516, 83311, 215646, 234721, 338409],
        "education": [
            "Bachelors",
            "Bachelors",
            "HS-grad",
            "11th",
            "Bachelors"
        ],
        "education-num": [13, 13, 9, 7, 13],
        "marital-status": [
            "Never-married",
            "Married-civ-spouse",
            "Divorced",
            "Married-civ-spouse",
            "Married-civ-spouse"
        ],
        "occupation": [
            "Adm-clerical",
            "Exec-managerial",
            "Handlers-cleaners",
            "Handlers-cleaners",
            "Prof-specialty"
        ],
        "relationship": [
            "Not-in-family",
            "Husband",
            "Not-in-family",
            "Husband",
            "Wife"
        ],
        "race": ["White", "White", "White", "Black", "Black"],
        "sex": ["Male", "Male", "Male", "Male", "Female"],
        "capital-gain": [2174, 0, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40, 40],
        "native-country": [
            "United-States",
            "United-States",
            "United-States",
            "United-States",
            "Cuba"
        ],
        "salary": ["<=50K", "<=50K", "<=50K", "<=50K", ">50K"]
    }
    return pd.DataFrame(data)


def test_process_data_output_shape(sample_data):
    """
    Tests that the process_data function returns X and y with the same
    number of rows.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]

    X, y, _, _, _ = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert len(X) == len(y)
    assert len(X) == 5


def test_train_model_returns_model_object():
    """
    Tests that the train_model function returns a scikit-learn model
    object.
    """
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, 10)

    model = train_model(X_train, y_train)

    assert isinstance(model, LogisticRegression)


def test_compute_model_metrics_perfect_score():
    """
    Tests that compute_model_metrics returns perfect scores for a
    perfect prediction.
    """
    y_true = np.array([1, 1, 0, 0])
    y_preds = np.array([1, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
