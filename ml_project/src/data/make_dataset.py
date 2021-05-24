import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entities import SplitParams

logger = logging.getLogger(__name__)


def read_data(path: str) -> pd.DataFrame:
    """
    Read data from csv file.
    :param path: path to the csv file
    :return: pandas dataframe with read data
    """
    data = pd.read_csv(path)
    logger.info(f"data.shape is {data.shape}")
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplitParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation datasets.
    :param data: original dataframe
    :param params: splitting parameters
    :return: tuple of training and validation dataframes
    """
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    logger.info(f"train_data.shape is {train_data.shape}")
    logger.info(f"val_data.shape is {val_data.shape}")
    return train_data, val_data
