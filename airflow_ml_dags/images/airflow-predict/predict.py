import os
import pickle

import click
import pandas as pd


DATA_FILENAME = "data.csv"
MODEL_FILENAME = "model.pkl"
PREDICTIONS_FILENAME = "predictions.csv"
TARGET_COL = "target"


@click.command("predict")
@click.option("--input_dir", required=True)
@click.option("--models_dir", required=True)
@click.option("--output_dir", required=True)
def predict(input_dir: str, models_dir: str, output_dir: str):
    data_path = os.path.join(input_dir, DATA_FILENAME)
    model_path = os.path.join(models_dir, MODEL_FILENAME)
    predictions_path = os.path.join(output_dir, PREDICTIONS_FILENAME)

    with open(model_path, "rb") as fin:
        model = pickle.load(fin)

    df = pd.read_csv(data_path)
    predictions = pd.DataFrame(model.predict(df))

    os.makedirs(output_dir, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)


if __name__ == "__main__":
    predict()
