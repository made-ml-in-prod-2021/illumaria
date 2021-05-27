import logging
import pickle
from typing import Dict, Optional, Union, NoReturn

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.entities.train_params import TrainParams

logger = logging.getLogger(__name__)

ClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainParams
) -> ClassificationModel:
    """
    Train the model.
    :param features: features to train on
    :param target: target labels
    :param train_params: training parameters
    :return: trained model class
    """
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            solver="liblinear", random_state=train_params.random_state
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    logger.info("Model successfully fitted.")
    return model


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    """
    Evaluate model and return the metrics.
    :param predicts: predicted labels
    :param target: target labels
    :return: a dict of type {'metric': value}
    """
    metrics = {
        "accuracy": accuracy_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }
    logger.info(f"Metrics are: {metrics}")
    return metrics


def serialize_model(model: ClassificationModel, path: str, transformer: Optional[ColumnTransformer] = None) -> NoReturn:
    """
    Save model to pickle file.
    :param transformer: the transformer to save
    :param model: the model to save
    :param path: the file to save to
    :return: the path to saved file
    """
    pipeline = Pipeline((
        [
            ("transformer", transformer),
            ("model", model),
        ]
    ))
    with open(path, "wb") as fout:
        pickle.dump(pipeline, fout)
    logger.info(f"Pipeline saved to {path}")
    return path
