# coding=utf-8
import numpy as np
from math import sqrt
from numpy import nanmean
from numpy.core._multiarray_umath import square


def rms(a):
    return sqrt(nanmean(square(a.flatten())))


def rmse(a):
    return sqrt(nanmean(square(a.flatten() - np.nanmean(a.flatten()))))


def pv(a):
    return np.nanmax(a.flatten()) - np.nanmin(a.flatten())


def nbTermsPoly(startDeg, endDeg):
    return nbTerms(endDeg) - nbTerms(startDeg)


def nbTerms(deg):
    return int((deg + 2) * (deg + 1) / 2)


def nbTermsLegendre2D(deg):
    return (deg + 1) ** 2
