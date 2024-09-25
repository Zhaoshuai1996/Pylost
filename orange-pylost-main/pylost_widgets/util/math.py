# coding=utf-8
import numpy as np
from astropy.units import Quantity
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def split_value_unit(a):
    if isinstance(a, Quantity):
        return a.value, a.unit
    else:
        return a, 1


def mad(a):
    """
    Mean absolute deviation.

    :param a: data array
    :type a: np.ndarray
    :return: mad of array
    :rtype: Quantity[float]
    """
    a, unit = split_value_unit(a)
    x = a[np.isfinite(a)]
    return (np.nansum(np.abs(x - np.nanmean(x))) / len(x)) * unit


def mask_outliers(a, threshold, center='median'):
    """
    Mask outlier pixels in an array beyond the threshold.

    :param a: Data array
    :type a: np.ndarray
    :param threshold: Threshold beyond which are set to nan
    :type threshold: float
    :param center: center method: median or mean
    :type center: str
    :return: Masked array
    :rtype: Quantity[np.ndarray]
    """
    a, unit = split_value_unit(a)
    if not np.isfinite(threshold):
        return a * unit
    if a is None or not np.any(a):
        return a * unit

    if center == 'median':
        return a[np.fabs(a - np.nanmedian(a)) < threshold] * unit
    elif center == 'mean':
        return a[np.fabs(a - np.nanmean(a)) < threshold] * unit
    else:
        return a * unit


def rms(a):
    """
    Root mean square value excluding nan's in an array (or astropy quantity array).

    :param a: Numpy array data or Quantity data
    :type a: np.ndarray
    :return: Rms of array
    :rtype: Quantity[float]
    """
    a, unit = split_value_unit(a)
    return np.sqrt(np.nanmean(np.square(a.ravel()))) * unit


def rmse(a):
    """
    Root mean square error value excluding nan's in an array (or astropy quantity array).

    :param a: Numpy array data or Quantity data
    :type a: np.ndarray
    :return: Rms error of array
    :rtype: Quantity[float]
    """
    a, unit = split_value_unit(a)
    return np.sqrt(np.nanmean(np.square(a.ravel() - np.nanmean(a.ravel())))) * unit


def pv(a):
    """
    Peak to valley value excluding nan's in an array (or astropy quantity array).

    :param a: Numpy array data or Quantity data
    :type a: np.ndarray
    :return: PV of array
    :rtype: Quantity[float]
    """
    a, unit = split_value_unit(a)
    return (np.nanmax(a.ravel()) - np.nanmin(a.ravel())) * unit


def nanstd(a, **kwargs):
    """
    Apply nanstd to values if array is quantity for faster calculation.

    :param a: Input numpy array
    :type a: np.ndarray
    :param kwargs: Additional arguments
    :type kwargs: dict
    :return: nanstd of array
    :rtype: Quantity[float]
    """
    a, unit = split_value_unit(a)
    return np.nanstd(a, **kwargs) * unit


def nanmean(a, **kwargs):
    """
    Apply nanmean to values if array is quantity for faster calculation.

    :param a: Input numpy array
    :type a: np.ndarray
    :param kwargs: Additional arguments
    :type kwargs: dict
    :return: nanmean of array
    :rtype: Quantity[float]
    """
    a, unit = split_value_unit(a)
    return np.nanmean(a, **kwargs) * unit


def nanmin(a, **kwargs):
    """
    Apply nanmin to values if array is quantity for faster calculation.

    :param a: Input numpy array
    :type a: np.ndarray
    :param kwargs: Additional arguments
    :type kwargs: dict
    :return: nanmin of array
    :rtype: Quantity[float]
    """
    a, unit = split_value_unit(a)
    return np.nanmin(a, **kwargs) * unit


def nanmax(a, **kwargs):
    """
    Apply nanmax to values if array is quantity for faster calculation.

    :param a: Input numpy array
    :type a: np.ndarray
    :param kwargs: Additional arguments
    :type kwargs: dict
    :return: nanmax of array
    :rtype: Quantity[float]
    """
    a, unit = split_value_unit(a)
    return np.nanmax(a, **kwargs) * unit


def nanmedian(a, **kwargs):
    """
    Apply nanmedian to values if array is quantity for faster calculation.

    :param a: Input numpy array
    :type a: np.ndarray
    :param kwargs: Additional arguments
    :type kwargs: dict
    :return: nanmedian of array
    :rtype: Quantity[float]
    """
    a, unit = split_value_unit(a)
    return np.nanmedian(a, **kwargs) * unit
