from typing import NoReturn, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

Array = Union[np.ndarray, pd.DataFrame]
Vector = Union[np.ndarray, pd.Series]


class SqrtTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for numerical features.
    """
    def __init__(self, copy: bool = True) -> NoReturn:
        super(__class__, self).__init__()

    def fit(self, x: Array, y: Vector = None) -> "SqrtTransformer":
        """
        Fit data.
        :param x: input data to fit
        :param y: target labels
        :return: fitted data
        """
        return self

    def transform(self, x: Array, y: Vector = None) -> Array:
        """
        Transform data.
        :param X: data to transform
        :param y: target labels
        :return: transformed data
        """
        x_copy = x.copy()
        x_copy = np.asarray(x_copy)
        x_copy -= np.min(x_copy, axis=0, keepdims=True)
        x_copy = np.sqrt(x_copy)
        return x_copy

    def fit_transform(self, x: Array, y: Vector = None) -> Array:
        """
        Fit and transform data.
        :param x: data to transform
        :param y: target labels
        :return: transformed data
        """
        return self.fit(x, y).transform(x, y)
