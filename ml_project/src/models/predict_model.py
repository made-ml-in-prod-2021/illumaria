import pickle

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def deserialize_model(path: str) -> Pipeline:
    """
    Load model from pickle file.
    :param path: file to load from
    :return: deserialized Pipeline class
    """
    with open(path, "rb") as fin:
        return pickle.load(fin)


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """
    Make predictions with model.
    :param model: the model class to predict with
    :param features: the features to predict on
    :return: model predictions
    """
    return model.predict(features)
