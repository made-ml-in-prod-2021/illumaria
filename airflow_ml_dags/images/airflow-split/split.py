import os

import click
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"
TRAIN_DATA_FILENAME = "train.csv"
VALID_DATA_FILENAME = "valid.csv"
TARGET_COL = "target"


@click.command("split")
@click.option("--input_dir", required=True)
@click.option("--output_dir", required=True)
@click.option("--train_size", default=0.8)
@click.option("--random_state", default=4)
def split(input_dir: str, output_dir: str, train_size: float, random_state: int):
    """
    Split data to train and validation parts.
    :param input_dir: path to preprocessed data
    :param output_dir: path to splitted data
    """
    input_path_data = os.path.join(input_dir, DATA_FILENAME)
    input_path_target = os.path.join(input_dir, TARGET_FILENAME)

    data = pd.read_csv(input_path_data)
    target = pd.read_csv(input_path_target)
    data[TARGET_COL] = target.values
    train_data, valid_data = train_test_split(
        data, train_size=train_size, random_state=random_state,
    )

    os.makedirs(output_dir, exist_ok=True)
    train_data_path = os.path.join(output_dir, TRAIN_DATA_FILENAME)
    valid_data_path = os.path.join(output_dir, VALID_DATA_FILENAME)
    train_data.to_csv(train_data_path, index=False)
    valid_data.to_csv(valid_data_path, index=False)


if __name__ == "__main__":
    split()
