"""Error metrics used in evaluation process.

The :mod:`errors` module contains a set of different error metrics
that is used for evaluating programmes during optimzation process.
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def sum_of_difference(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Sum of difference error metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    return np.sum(np.abs(y_pred - y_true))


def mean_squared_error_c(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Mean square error metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    return mean_squared_error(y_true, y_pred)


def root_mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Root mean square error metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    return mean_squared_error(y_true, y_pred, squared=False)


def mean_absolute_error_c(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Mean absolute error metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    return mean_absolute_error(y_true, y_pred)


def r2_score_inverse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Inverse of r2 Score metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    return 1 - r2_score(y_true, y_pred)
