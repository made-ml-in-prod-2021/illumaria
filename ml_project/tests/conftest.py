import os
from typing import List

import pytest


@pytest.fixture(scope="module")
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture(scope="module")
def model_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "model.pkl")


@pytest.fixture(scope="module")
def output_data_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "prediction.csv")


@pytest.fixture(scope="module")
def target_col():
    return "target"


@pytest.fixture(scope="module")
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture(scope="module")
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]


@pytest.fixture(scope="module")
def features_to_drop() -> List[str]:
    return []
