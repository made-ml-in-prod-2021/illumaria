import json
import os
import pickle

import click
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

VALID_FILENAME = "valid.csv"
MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.json"
TARGET_COL = "target"


@click.command("validate")
@click.option("--input_dir")
@click.option("--models_dir")
def validate(input_dir: str, models_dir: str):
    """
    Validate the model.
    :param input_dir: path to validation data
    :param models_dir: path to load the model and save the metrics
    """
    valid_data_path = os.path.join(input_dir, VALID_FILENAME)
    model_path = os.path.join(models_dir, MODEL_FILENAME)
    metric_path = os.path.join(models_dir, METRICS_FILENAME)

    valid_data = pd.read_csv(valid_data_path)
    with open(model_path, "rb") as fin:
        model = pickle.load(fin)
    
    predictions = model.predict(valid_data.drop(columns=TARGET_COL, axis=1))

    metrics = {
        "accuracy": accuracy_score(valid_data[TARGET_COL], predictions),
        "f1_score": f1_score(valid_data[TARGET_COL], predictions, average="macro"),
    }

    with open(metric_path, "w") as fout:
        json.dump(metrics, fout)


if __name__ == "__main__":
    validate()
