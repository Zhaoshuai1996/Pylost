# coding=utf-8
import numpy as np
from numpy import sqrt as sqrt


def nbTerms_LZ(deg):
    return int((deg + 2) * (deg + 1) / 2)


def zernike_xy(x, y, P=None, ignore_terms=[], start_degree=0, degree=7, coef=None):
    '''
    Lakshminarayanan, Vasudevan, and Andre Fleck. "Zernike polynomials: a guide." Journal of Modern Optics 58.7 (2011): 545-561.•
    :param x: 
    :param y: 
    :param P: 
    :return: 
    '''''

    if x.shape != y.shape:
        return None

    Zc = 0
    if degree > 7:
        degree = 7
    cnt = nbTerms_LZ(degree)
    if P is None or not any(P):
        P = [1] * cnt
    if start_degree > 0:
        ignore_terms += list(np.arange(nbTerms_LZ(start_degree - 1))) if start_degree > 0 else []
    for term in ignore_terms:
        P[term] = 0
    Z = np.full(x.shape + (cnt,), np.nan)

    Znm = {}
    Znm[(0, 0)] = Z[..., 0] = P[0] * sqrt(2) * 1
    if degree >= 1:
        Znm[(1, -1)] = Z[..., 1] = P[1] * sqrt(4) * x
        Znm[(1, 1)] = Z[..., 2] = P[2] * sqrt(4) * y
    if degree >= 2:
        Znm[(2, -2)] = Z[..., 3] = P[3] * sqrt(6) * 2 * x * y
        Znm[(2, 0)] = Z[..., 4] = P[4] * sqrt(3) * (-1 + 2 * x ** 2 + 2 * y ** 2)
        Znm[(2, 2)] = Z[..., 5] = P[5] * sqrt(6) * (-x ** 2 + y ** 2)
    if degree >= 3:
        Znm[(3, -3)] = Z[..., 6] = P[6] * sqrt(8) * (-x ** 3 + 3 * x * y ** 2)
        Znm[(3, -1)] = Z[..., 7] = P[7] * sqrt(8) * (-2 * x + 3 * x ** 3 + 3 * x * y ** 2)
        Znm[(3, 1)] = Z[..., 8] = P[8] * sqrt(8) * (-2 * y + 3 * y ** 3 + 3 * x ** 2 * y)
        Znm[(3, 3)] = Z[..., 9] = P[9] * sqrt(8) * (y ** 3 - 3 * x ** 2 * y)
    if degree >= 4:
        Znm[(4, -4)] = Z[..., 10] = P[10] * sqrt(10) * (-4 * x ** 3 * y + 4 * x * y ** 3)
        Znm[(4, -2)] = Z[..., 11] = P[11] * sqrt(10) * (-6 * x * y + 8 * x ** 3 * y + 8 * x * y ** 3)
        Znm[(4, 0)] = Z[..., 12] = P[12] * sqrt(5) * (
                1 - 6 * x ** 2 - 6 * y ** 2 + 6 * x ** 4 + 12 * x ** 2 * y ** 2 + 6 * y ** 4)
        Znm[(4, 2)] = Z[..., 13] = P[13] * sqrt(10) * (3 * x ** 2 - 3 * y ** 2 - 4 * x ** 4 + 4 * y ** 4)
        Znm[(4, 4)] = Z[..., 14] = P[14] * sqrt(10) * (x ** 4 - 6 * x ** 2 * y ** 2 + y ** 4)
    if degree >= 5:
        Znm[(5, -5)] = Z[..., 15] = P[15] * sqrt(12) * (x ** 5 - 10 * x ** 3 * y ** 2 + 5 * x * y ** 4)
        Znm[(5, -3)] = Z[..., 16] = P[16] * sqrt(12) * (
                4 * x ** 3 - 12 * x * y ** 2 - 5 * x ** 5 + 10 * x ** 3 * y ** 2 + 15 * x * y ** 4)
        Znm[(5, -1)] = Z[..., 17] = P[17] * sqrt(12) * (
                3 * x - 12 * x ** 3 - 12 * x * y ** 2 + 10 * x ** 5 + 20 * x ** 3 * y ** 2 + 10 * x * y ** 4)
        Znm[(5, 1)] = Z[..., 18] = P[18] * sqrt(12) * (
                3 * y - 12 * y ** 3 - 12 * x ** 2 * y + 10 * y ** 5 + 20 * x ** 2 * y ** 3 + 10 * x ** 4 * y)
        Znm[(5, 3)] = Z[..., 19] = P[19] * sqrt(12) * (
                -4 * y ** 3 + 12 * x ** 2 * y + 5 * y ** 5 - 10 * x ** 2 * y ** 3 - 15 * x ** 4 * y)
        Znm[(5, 5)] = Z[..., 20] = P[20] * sqrt(12) * (y ** 5 - 10 * x ** 2 * y ** 3 + 5 * x ** 4 * y)
    if degree >= 6:
        Znm[(6, -6)] = Z[..., 21] = P[21] * sqrt(14) * (6 * x ** 5 * y - 20 * x ** 3 * y ** 3 + 6 * x * y ** 5)
        Znm[(6, -4)] = Z[..., 22] = P[22] * sqrt(14) * (
                20 * x ** 3 * y - 20 * x * y ** 3 - 24 * x ** 5 * y + 24 * x * y ** 5)
        Znm[(6, -2)] = Z[..., 23] = P[23] * sqrt(14) * (
                12 * x * y - 40 * x ** 3 * y - 40 * x * y ** 3 + 30 * x ** 5 * y + 60 * x ** 3 * y ** 3 - 30 * x * y ** 5)
        Znm[(6, 0)] = Z[..., 24] = P[24] * sqrt(7) * (
                -1 + 12 * x ** 2 + 12 * y ** 2 - 30 * x ** 4 - 60 * x ** 2 * y ** 2 - 30 * y ** 4 + 20 * x ** 6 + 60 * x ** 4 * y ** 2 + 60 * x ** 2 * y ** 4 + 20 * y ** 6)
        Znm[(6, 2)] = Z[..., 25] = P[25] * sqrt(14) * (
                -6 * x ** 2 + 6 * y ** 2 + 20 * x ** 4 - 20 * y ** 4 - 15 * x ** 6 - 15 * x ** 4 * y ** 2 + 15 * x ** 2 * y ** 4 + 15 * y ** 6)
        Znm[(6, 4)] = Z[..., 26] = P[26] * sqrt(14) * (
                -5 * x ** 4 + 30 * x ** 2 * y ** 2 - 5 * y ** 4 + 6 * x ** 6 - 30 * x ** 4 * y ** 2 - 30 * x ** 2 * y ** 4 + 6 * y ** 6)
        Znm[(6, 6)] = Z[..., 27] = P[27] * sqrt(14) * (-x ** 6 + 15 * x ** 4 * y ** 2 - 15 * x ** 2 * y ** 4 + y ** 6)
    if degree >= 7:
        Znm[(7, -7)] = Z[..., 28] = P[28] * sqrt(16) * (
                -x ** 7 + 21 * x ** 5 * y ** 2 - 35 * x ** 3 * y ** 4 + 7 * x * y ** 6)
        Znm[(7, -5)] = Z[..., 29] = P[29] * sqrt(16) * (
                -6 * x ** 5 + 60 * x ** 3 * y ** 2 - 30 * x * y ** 4 + 7 * x ** 7 - 63 * x ** 5 * y ** 2 - 35 * x ** 3 * y ** 4 + 35 * x * y ** 6)
        Znm[(7, -3)] = Z[..., 30] = P[30] * sqrt(16) * (
                -10 * x ** 3 + 30 * x * y ** 2 + 30 * x ** 5 - 60 * x ** 3 * y ** 2 - 90 * x * y ** 4 - 21 * x ** 7 + 21 * x ** 5 * y ** 2 + 105 * x ** 3 * y ** 4 + 63 * x * y ** 6)
        Znm[(7, -1)] = Z[..., 31] = P[31] * sqrt(16) * (
                -4 * x + 30 * x ** 3 + 30 * x * y ** 2 - 60 * x ** 5 - 120 * x ** 3 * y ** 2 - 60 * x * y ** 4 + 35 * x ** 7 + 105 * x ** 5 * y ** 2 + 105 * x ** 3 * y ** 4 + 35 * x * y ** 6)
        Znm[(7, 1)] = Z[..., 32] = P[32] * sqrt(16) * (
                -4 * y + 30 * y ** 3 + 30 * x ** 2 * y - 60 * y ** 5 - 120 * x ** 2 * y ** 3 - 60 * x ** 4 * y + 35 * y ** 7 + 105 * x ** 2 * y ** 5 + 105 * x ** 4 * y ** 3 + 35 * x ** 6 * y)
        Znm[(7, 3)] = Z[..., 33] = P[33] * sqrt(16) * (
                10 * y ** 3 - 30 * x ** 2 * y - 30 * y ** 5 + 60 * x ** 2 * y ** 3 + 90 * x ** 4 * y + 21 * y ** 7 - 21 * x ** 2 * y ** 5 - 105 * x ** 4 * y ** 3 + 63 * x ** 6 * y)
        Znm[(7, 5)] = Z[..., 34] = P[34] * sqrt(16) * (
                -6 * y ** 5 + 60 * x ** 2 * y ** 3 - 30 * x ** 4 * y + 7 * y ** 7 - 63 * x ** 2 * y ** 5 - 35 * x ** 4 * y ** 3 + 35 * x ** 6 * y)
        Znm[(7, 7)] = Z[..., 35] = P[35] * sqrt(16) * (
                y ** 7 - 21 * x ** 2 * y ** 5 + 35 * x ** 4 * y ** 3 - 7 * x ** 6 * y)

    if coef is not None and len(coef) == len(Znm):
        Zr = list(Znm.values())
        for i, c in enumerate(coef):
            Zc += c * Zr[i]
    return Z, Znm, Zc


def legendre_xy(x, y, P=None, norm=True, ignore_terms=[], start_degree=0, degree=8, coef=None):
    '''
    Lakshminarayanan, Vasudevan, and Andre Fleck. "Zernike polynomials: a guide." Journal of Modern Optics 58.7 (2011): 545-561.•
    :param x: 
    :param y: 
    :param P: 
    :return: 
    '''''

    if x.shape != y.shape:
        return None

    Zc = 0
    if degree > 8:
        degree = 8
    cnt = nbTerms_LZ(degree)
    if P is None or not any(P):
        P = [1] * cnt
    if start_degree > 0:
        ignore_terms += list(np.arange(nbTerms_LZ(start_degree - 1))) if start_degree > 0 else []
    for term in ignore_terms:
        P[term] = 0

    Z = np.full(x.shape + (cnt,), np.nan)

    Znm = {}
    j = 0
    for deg in np.arange(degree + 1):
        for i in np.arange(deg + 1):
            Znm[(deg - i, i)] = Z[..., j] = P[j] * np.multiply(legendre_1D(deg - i, x, norm), legendre_1D(i, y, norm))
            j += 1

    if coef is not None and len(coef) == len(Znm):
        Zr = list(Znm.values())
        for i, c in enumerate(coef):
            Zc += c * Zr[i]
    return Z, Znm, Zc


def legendre_1D(Ln, X, norm=True):
    """
    ------ Rafael Celestre -----
    Calculates the 1D Legendre polynomials on a X grid ranging from -1 to 1. The polynomials can be obtained using the
    Rodrigues formula (https://en.wikipedia.org/wiki/Rodrigues%27_formula): Ln = 1/(2^n n!) * d^n(x^2-1)^n/dx^n, where
    d^n/dx^n indicates the nth derivative of (x^2-1)^n

    @param [in] Ln: polynomial index
    @param [in] X: grid ranging from [-1,1]. np.linspace(-1,1,npix) or np.meshgrid(-1,1,npix)
    @param [in] norm: puts the normal in orthonormal, otherwise the base is just orthogonal
    @return 1D legendre polynomial calculated over a grid.
    """

    k = 1
    if norm is True:
        k = np.sqrt(2 * Ln + 1)

    if Ln == 0:  # Piston
        return np.ones(X.shape) * k
    elif Ln == 1:  # Tilt
        return X
    elif Ln == 2:  # Defocus
        return (3 * X ** 2 - 1) / 2
    elif Ln == 3:  # Coma
        return (5 * X ** 3 - 3 * X) / 2
    elif Ln == 4:  # Spherical aberration
        return (35 * X ** 4 - 30 * X ** 2 + 3) / 8
    elif Ln == 5:  # Secondary coma
        return (63 * X ** 5 - 70 * X ** 3 + 15 * X) / 8
    elif Ln == 6:  # Secondary spherical aberration
        return (231 * X ** 6 - 315 * X ** 4 + 105 * X ** 2 - 5) / 16
    elif Ln == 7:  # Tertiary coma
        return (429 * X ** 7 - 693 * X ** 5 + 315 * X ** 3 - 35 * X) / 16
    elif Ln == 8:  # Tertiary spherical aberration
        return (6435 * X ** 8 - 12012 * X ** 6 + 6930 * X ** 4 - 1260 * X ** 2 + 35) / 128
