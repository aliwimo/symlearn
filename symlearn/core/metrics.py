"""Error metrics used in evaluation process.

The :mod:`errors` module contains a set of different error metrics
that is used for evaluating programmes during optimization process.
"""
import numpy as np


def sum_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Sum of absolute error metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    return np.sum(np.abs(y_pred - y_true))


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Mean square error metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred)**2)


def root_mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Root mean square error metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Mean absolute error metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_pred, y_true):
    """R2 Score metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)


def r2_score_inverse(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Inverse of R2 Score metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    return 1 - r2_score(y_pred, y_true)
    

def max_residual_error(y_true, y_pred):
    """Maximum residual error metric

    Args:
        y_pred (np.ndarray): Predicted output values array
        y_true (np.ndarray): Real output values array

    Returns:
        The value of the metric
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.max(np.abs(y_true - y_pred))