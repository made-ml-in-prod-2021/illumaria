import os
from shutil import copyfile

import click
import pandas as pd

DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"


@click.command("preprocess_data")
@click.option("--input_dir")
@click.option("--output_dir")
def preprocess(input_dir: str, output_dir: str):
    """
    Preprocess train data.
    :param input_dir: path to raw train data
    :param output_dir: path to processed train data
    """
    raw_data_path = os.path.join(input_dir, DATA_FILENAME)
    raw_target_path = os.path.join(input_dir, TARGET_FILENAME)

    os.makedirs(output_dir, exist_ok=True)
    processed_data_path = os.path.join(output_dir, DATA_FILENAME)
    processed_target_path = os.path.join(output_dir, TARGET_FILENAME)

    train_df = pd.read_csv(raw_data_path)
    train_df.fillna(0)
    train_df.to_csv(processed_data_path, index=False)

    if os.path.isfile(raw_target_path):
        copyfile(raw_target_path, processed_target_path)


if __name__ == "__main__":
    preprocess()
