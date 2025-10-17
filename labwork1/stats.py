import numpy as np


def mean(x: np.ndarray) -> float:
    return np.sum(x) / x.size


def variance(x: np.ndarray) -> float:
    m = mean(x)
    return np.sum(x**2 - m**2) / x.size


def covariance(x: np.ndarray, y: np.ndarray) -> float:
    mx = mean(x)
    my = mean(y)
    return np.sum((x * y) - (mx * my)) / x.size


def regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    a = covariance(x, y) / variance(x)
    b = mean(y) - a * mean(x)

    y = a * x + b
    return y


def median(x: np.ndarray) -> float:
    sorted_x = np.sort(x)
    n = x.size
    mid = n // 2

    if n % 2 == 0:
        return (sorted_x[mid - 1] + sorted_x[mid]) / 2
    else:
        return sorted_x[mid]
