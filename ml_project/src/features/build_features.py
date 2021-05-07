from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.entities.feature_params import FeatureParams


def build_categorical_pipeline() -> Pipeline:
    """
    Build pipeline for categorical features processing.
    :return: pipeline class
    """
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown='ignore')),
        ]
    )
    return categorical_pipeline


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build pipeline and process categorical features.
    :param categorical_df: dataframe of categorical features
    :return: processed dataframe
    """
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_numerical_pipeline() -> Pipeline:
    """
    Build pipeline for numerical features processing.
    :return: pipeline class
    """
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")), ]
    )
    return num_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build pipeline and process numerical features.
    :param categorical_df: dataframe of numerical features
    :return: processed dataframe
    """
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    """
    Build transformer to process both categorical and numerical features.
    :param params: configuration for features processing
    :return: transformer class
    """
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer


def make_features(
    transformer: ColumnTransformer,
    df: pd.DataFrame,
    params: FeatureParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process features.
    :param transformer: transformer class to process features
    :param df: features dataframe
    :param params: configuration for features processing
    :return: processed dataframes of features and targets
    """
    target = df[params.target_col] if params.target_col in df.columns else None
    return pd.DataFrame(transformer.transform(df)), target
