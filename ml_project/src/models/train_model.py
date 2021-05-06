import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src.entities.train_params import TrainingParams

SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassificationModel:
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
    return model


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts),
        "roc_auc": roc_auc_score(target, predicts),
    }


def serialize_model(model: SklearnClassificationModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
