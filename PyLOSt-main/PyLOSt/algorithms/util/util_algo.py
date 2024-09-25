# coding=utf-8
from astropy.units import Quantity
from astropy import units as u

from PyLOSt.algorithms.util.util_math import rms
from PyLOSt.databases.gs_table_classes import ConfigParams
import numpy as np

from PyLOSt.algorithms.util.util_fit import getPixSz2D, fit2D


def get_default_data_names():
    qdef_names = ConfigParams.selectBy(paramName='DEFAULT_DATA_NAMES')[0]
    return qdef_names.paramValue.split(',')


def differentiate_heights(z, x=None, y=None, pix_sz=1, method='grad'):
    pix_sz = getPixSz2D(pix_sz)
    sx = None
    sy = None
    if method == 'grad':
        dx = pix_sz[0] if x is None else x
        dy = pix_sz[1] if y is None else y
        if isinstance(dx, Quantity):
            sx = np.gradient(z, dx.value, axis=-1) / dx.unit
        else:
            sx = np.gradient(z, dx, axis=-1)
        if isinstance(dy, Quantity):
            sy = np.gradient(z, dy.value, axis=-2) / dy.unit if z.ndim >= 2 else None
        else:
            sy = np.gradient(z, dy, axis=-2) if z.ndim >= 2 else None
    elif method == 'diff':
        dx = pix_sz[0] if x is None else np.diff(x)
        dy = pix_sz[1] if y is None else np.diff(y)
        sx = np.divide(np.diff(z, axis=-1), dx)
        sy = np.divide(np.diff(z, axis=-2), dy) if z.ndim >= 2 else None

    if z.__class__.__name__ == 'MetrologyData':
        return sx * u.rad, sy * u.rad
    else:
        return sx, sy


def filtBadPixels(self, data, pix_sz):
    """
    Filter pixels above n-std from global shape.

    :param self: Stitching function reference object
    :param sarr: Input data
    :param pix_sz: Pixel size
    :return: Filter mask
    """
    maxStd = float(self.filtBadPixMaxStd)
    f1, f1resd, _ = fit2D(data, pix_size=pix_sz, degree=2, retResd=True)
    rmsResd = rms(f1resd)
    maskBadPix = f1resd < maxStd * rmsResd
    return maskBadPix
