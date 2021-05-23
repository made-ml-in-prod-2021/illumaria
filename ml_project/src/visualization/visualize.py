import argparse
from pathlib import Path
from typing import NoReturn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.entities.project_params import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    LABEL_COLUMN,
)
from src.utils import setup_logger

logger = setup_logger(path="logs/visualize.log")


def save_statistics(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    """
    Save the data info and value statistics.
    :param data: raw data to process
    :param output_dir: directory to store the info
    :return: nothing
    """
    description = data.describe()
    description.loc["nunique", :] = data.nunique()
    description.loc["dtype", :] = data.dtypes
    description = description.round(3)
    description.to_csv(output_dir / "statistics.csv")


def save_categorical_plots(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    """
    Save the plots for categorical column values.
    :param data: raw data to process
    :param output_dir: directory to store the figure
    :return: nothing
    """
    plt.figure(figsize=(12, 3))
    sns.countplot(x=data[LABEL_COLUMN])
    plt.grid()
    plt.savefig(output_dir / f"plot_{LABEL_COLUMN}.png")

    for column in CATEGORICAL_COLUMNS:
        plt.figure(figsize=(12, 3))
        sns.countplot(x=data[column], hue=data[LABEL_COLUMN])
        plt.legend(loc="best", title="target", fontsize=12)
        plt.grid()
        plt.savefig(output_dir / f"plot_cat_{column}.png")


def save_numerical_plots(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    """
    Save the plots for numerical column values.
    :param data: raw data to process
    :param output_dir: directory to store the figure
    :return: nothing
    """
    for column in NUMERICAL_COLUMNS:
        plt.figure(figsize=(12, 3))
        sns.boxplot(x=data[column], y=data[LABEL_COLUMN], orient="h")
        plt.grid()
        plt.savefig(output_dir / f"plot_num_{column}.png")


def save_pairplot(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    """
    Save the pairplot for numerical column values and target values.
    :param data: raw data to process
    :param output_dir: directory to store the figure
    :return: nothing
    """
    plt.figure(figsize=(16, 16))
    sns.pairplot(data[list(NUMERICAL_COLUMNS.keys()) + [LABEL_COLUMN]], hue=LABEL_COLUMN)
    plt.savefig(output_dir / "pairplot.png")


def save_heatmap(data: pd.DataFrame, output_dir: Path) -> NoReturn:
    """
    Save the cross-correlation heatmap for column values.
    :param data: raw data to process
    :param output_dir: directory to store the figure
    :return: nothing
    """
    plt.figure(figsize=(16, 16))
    sns.heatmap(data.corr(), cmap="summer", annot=True, linewidths=0.5)
    plt.savefig(output_dir / "heatmap.png")


def setup_parser(parser: argparse.ArgumentParser) -> NoReturn:
    """The function to setup parser arguments."""
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="the path to the input data",
        required=True,
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="the folder to save reports to",
        required=True,
    )


def main():
    """
    The main module function to parse arguments and generate reports.
    :return: nothing
    """
    parser = argparse.ArgumentParser(
        prog="report-generator",
        description="A tool to generate reports on the data provided.",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = list(CATEGORICAL_COLUMNS.keys())
    required_cols.extend(NUMERICAL_COLUMNS.keys())
    required_cols.append(LABEL_COLUMN)

    data = pd.read_csv(Path(arguments.input), usecols=required_cols)

    logger.info("Saving statictics...")
    save_statistics(data, output_dir)
    logger.info("Saving categorical plots...")
    save_categorical_plots(data, output_dir)
    logger.info("Saving numerical plots...")
    save_numerical_plots(data, output_dir)
    logger.info("Saving pairplot...")
    save_pairplot(data, output_dir)
    logger.info("Saving heatmap...")
    save_heatmap(data, output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
