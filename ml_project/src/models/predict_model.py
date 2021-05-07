import pickle
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def deserialize_model(path: str) -> ClassificationModel:
    """
    Load model from pickle file.
    :param path: file to load from
    :return: deserialized model class
    """
    with open(path, "rb") as fin:
        return pickle.load(fin)


def predict_model(model: ClassificationModel, features: pd.DataFrame) -> np.ndarray:
    """
    Make predictions with model.
    :param model: the model class to predict with
    :param features: the features to predict on
    :return: model predictions
    """
    return model.predict(features)
