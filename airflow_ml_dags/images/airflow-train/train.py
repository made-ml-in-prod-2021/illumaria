import os
import pickle

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TRAIN_FILENAME = "train.csv"
MODEL_FILENAME = "model.pkl"
TARGET_COL = "target"


@click.command("train")
@click.option("--input_dir")
@click.option("--models_dir")
@click.option("--random_state", default=4)
def train(input_dir: str, models_dir: str, random_state: int):
    """
    Train the model.
    :param input_dir: path to train data
    :param models_dir: path to save the model artifacts
    :param random_state: model random state
    """
    train_data_path = os.path.join(input_dir, TRAIN_FILENAME)
    model_save_path = os.path.join(models_dir, MODEL_FILENAME)

    train_df = pd.read_csv(train_data_path)
    model = RandomForestClassifier(random_state=random_state)
    model.fit(train_df.drop(columns=TARGET_COL, axis=1), train_df[TARGET_COL])

    os.makedirs(models_dir, exist_ok=True)
    with open(model_save_path, "wb") as fout:
        pickle.dump(model, fout)


if __name__ == "__main__":
    train()
