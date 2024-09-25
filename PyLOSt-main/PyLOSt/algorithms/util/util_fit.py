# coding=utf-8

import numpy as np
import scipy
from scipy.optimize import curve_fit

from PyLOSt.algorithms.util.util_math import nbTermsPoly, pv
from PyLOSt.algorithms.util.util_poly import legendre_xy, zernike_xy


def fit1D(Z, pix_size=[1, 1], mask=None, method='LS', degree=1, start_degree=0, typ='poly', dtyp='height',
          filter_terms_poly=[],
          elp_params=[], retResd=True, x=None, **kwargs):
    """
    Fit a polynomial to 1D data

    :param Z: 1D data
    :param pix_size: Pixel size in mm
    :param mask: Data mask
    :param method: Fit method
    :param degree: Polynomial degree
    :return: Coefficients, residuals
    """
    if degree == 0:
        return [], Z - np.nanmean(Z.flatten()), np.nanmean(Z.flatten())
    pix_size = getPixSz2D(pix_size)
    Z = Z.ravel()
    Zfit = None
    if not np.any(mask):
        mask = ~np.isnan(Z)
    nx = len(Z)

    if not np.any(x):
        x = np.arange(0, nx)
        x = pix_size[0] * (x - np.nanmean(x))
    else:
        x = x - np.nanmean(x)

    xf = x[mask]
    Zf = Z[mask]

    if typ == 'ellipse':  # Tangential ellipse only
        if len(elp_params) > 0:
            elp_params_check = np.asarray(kwargs['elp_params_check'], dtype=int) if 'elp_params_check' in kwargs else [
                0, 1, 1, 0, 0, 0]
            x0, z0 = calc_center_1d(x, Z)
            xf = xf - x0
            Zf = Zf - z0
            if kwargs.get('sel_script', 'EllipsePylost') != 'EllipsePylost':
                cls = kwargs.get('sel_script_class', None)
                if cls is not None:
                    Zscale = kwargs.get('Zscale', 1)
                    pix_scale = kwargs.get('pix_scale', [1, 1])
                    obj = cls(elp_params, elp_params_check)
                    C, _, _ = obj.fit(dtyp, Zf * Zscale, xf * pix_scale[-1])
                    if retResd:
                        Zfit = obj.get_ellipse(dtyp, x * pix_scale[-1], C)
                    # Rescale items
                    C[2] = C[2] * 1e3  # theta in mrad
                    C[3] = C[3] * 1e3  # center_offset in mm
                    C[4] = C[4] * 1e3  # rotate in mrad
                    C[5] = C[5] * 1e9  # piston in nm
                    Zfit = Zfit / Zscale
            else:
                err_func, fn_filt = filter_func(elp_params_check, dtyp, ftyp='ellipse', elp_params=elp_params, **kwargs)

                # C, _ = curve_fit(err_func, (fn_filt, xf), Zf, p0=elp_params)
                Cf = np.array(elp_params)[np.array(elp_params_check, dtype=bool)]
                Cnf = np.array(elp_params)[~np.array(elp_params_check, dtype=bool)]
                res = scipy.optimize.least_squares(err_func, x0=Cf, args=(Cnf, xf, Zf, fn_filt), loss='linear',
                                                   xtol=1e-4, ftol=1e-12, gtol=1e-15)
                C = get_C(res.x, Cnf, elp_params_check)
                if retResd:
                    Zfit = fn_filt(x, *C)
    else:  # polynomial by default
        if np.any(filter_terms_poly):
            degree = -1
        A = matPoly(xf, None, degree, start_degree, nbVar=1, terms=filter_terms_poly, dtyp=dtyp)
        C, _, _, _ = scipy.linalg.lstsq(A, Zf)  # coefficients C[0] + C[1]*X + C[2]*X2
        if retResd:
            Zfit = evalPoly(C, x, None, degree, start_degree, nbVar=1, terms=filter_terms_poly, dtyp=dtyp)

    resd = Z - Zfit if retResd else None
    return C, resd, Zfit


def fit_nD(Z, **kwargs):
    """

    :param Z:
    :param kwargs:
    :return:
    """
    dim_detector = kwargs['dim_detector'] if 'dim_detector' in kwargs else []
    axis_vals = kwargs['axis_vals'] if 'axis_vals' in kwargs else [[]] * Z.ndim
    if np.any(dim_detector) and False in dim_detector:  # Apply fit for selected detector dimensions.
        if not np.sum(dim_detector) in [1, 2]:  # only 1D or 2D
            return None, Z, None
        shp_nondetctr = tuple(np.asarray(Z.shape)[np.invert(dim_detector)])
        Zn = np.full_like(Z, np.nan)
        Coeff_nD = None
        Zn_fit = np.full_like(Z, np.nan)
        idx_full = np.asarray([slice(None)] * Z.ndim)
        for idx in np.ndindex(shp_nondetctr):
            idx_full[np.invert(dim_detector)] = np.asarray(idx)
            if np.sum(dim_detector) == 2:  # 2D surfaces
                xv, yv = getGrid(axis_vals[-1], axis_vals[-2])
                output = fit2D(Z[tuple(idx_full)], xv=xv, yv=yv, **kwargs)
            else:  # 1D lines
                output = fit1D(Z[tuple(idx_full)], x=axis_vals[-1], **kwargs)
            if Coeff_nD is None:
                Coeff_nD = np.full(shp_nondetctr + np.asarray(output[0]).shape, np.nan)
            Coeff_nD[idx], Zn[tuple(idx_full)], Zn_fit[tuple(idx_full)] = output

        return Coeff_nD, Zn, Zn_fit
    else:
        if Z.ndim == 1:
            return fit1D(Z, x=axis_vals[-1], **kwargs)
        elif Z.ndim == 2:
            xv, yv = getGrid(axis_vals[-1], axis_vals[-2])
            return fit2D(Z, xv=xv, yv=yv, **kwargs)
        else:
            Zn = np.full_like(Z, np.nan)
            Coeff_nD = None
            Zn_fit = np.full_like(Z, np.nan)
            for idx in np.ndindex(Z.shape[:-2]):
                xv, yv = getGrid(axis_vals[-1], axis_vals[-2])
                output = fit2D(Z[idx], xv=xv, yv=yv, **kwargs)
                if Coeff_nD is None:
                    Coeff_nD = np.full(Z.shape[:-2] + output[0].shape, np.nan)
                Coeff_nD[idx], Zn[idx], Zn_fit[idx] = output

            return Coeff_nD, Zn, Zn_fit


def getGrid(x, y):
    if not isinstance(x, (np.ndarray, list, tuple)):
        return None, None
    if not isinstance(y, (np.ndarray, list, tuple)):
        return None, None
    if len(x) == 0 or len(y) == 0:
        return None, None
    if not len(x) == len(y):
        return None, None

    return np.meshgrid(x - np.nanmean(x), y - np.nanmean(x))


def fit2D(Z, pix_size=[1, 1], mask=None, method='LS', degree=1, start_degree=0, retResd=True, useMaskForXY=True,
          typ='poly', dtyp='height', filter_terms_poly=[], elp_params=[], xv=None, yv=None, **kwargs):
    """
    Fit a polynomial to 2D data

    :param Z: Input data
    :param pix_size:  Pixel size
    :param mask: Data mask
    :param method: Fitting method e.g. least squares
    :param degree: Polynomial degree
    :param start_degree: Ignore polynomial coefficients below degree
    :param retResd: Flag return residuals
    :param useMaskForXY: Flag use a given mask or use mask on full data
    :param typ: Fit type e.g. polynomial, legendre etc...
    :param xv: X-grid positions
    :param yv: Y-grid positions
    :return: Fit coefficients, residuals
    """
    more_opts = kwargs.get('more_opts', {})
    if degree == 0:
        C = np.nanmean(Z.flatten())
        return np.array([C]), Z - C, C
    if not np.any(mask):
        mask = ~np.isnan(Z)

    if not (np.any(xv) and np.any(yv)):
        if useMaskForXY:
            xv, yv = getXYGrid(Z, pix_size, order=2, mask=mask)
        else:
            xv, yv = getXYGrid(Z, pix_size, order=2, mask=np.full(Z.shape, True))

    Zvf = Z
    xvf = xv
    yvf = yv
    if 'bin_fit' in more_opts and more_opts['bin_fit']:
        if more_opts['bin_x'] > 1 or more_opts['bin_y'] > 1:
            xvf = xv[::more_opts['bin_y'], ::more_opts['bin_x']]
            yvf = yv[::more_opts['bin_y'], ::more_opts['bin_x']]
            Zvf = Z[::more_opts['bin_y'], ::more_opts['bin_x']]
            mask = mask[::more_opts['bin_y'], ::more_opts['bin_x']]

    xf = xvf[mask]
    yf = yvf[mask]
    Zf = Zvf[mask]

    if 'center_data' in more_opts and more_opts['center_data']:
        x0, y0, z0 = calc_center_2d(xvf, yvf, Zvf)
        xf = xf - x0
        Zf = Zf - z0
    if 'rescale' in more_opts and more_opts['rescale'] and 'rescale_type' in more_opts:
        if more_opts['rescale_type'] == 0:  # Auto scale
            rxy = pv(np.asarray([xf, yf]))
            rz = pv(Zf)
            scale = rxy / rz
            Zf = Zf * scale

    A = None
    Zfit = None
    C = None
    if typ == 'ellipse':  # Tangential ellipse only
        if len(elp_params) > 0:
            elp_params_check = np.asarray(kwargs['elp_params_check'], dtype=int) if 'elp_params_check' in kwargs else [
                0, 1, 1, 0, 0, 0]
            if kwargs.get('sel_script', 'EllipsePylost') != 'EllipsePylost':
                cls = kwargs.get('sel_script_class', None)
                if cls is not None:
                    Zscale = kwargs.get('Zscale', 1)
                    pix_scale = kwargs.get('pix_scale', [1, 1])
                    elp_params_check[4] = more_opts['ellipse_rotation']  # workaround to pass auto rotation option
                    obj = cls(elp_params, elp_params_check)
                    xo = xvf[int(xvf.shape[-2] / 2), :] * pix_scale[-1]
                    yo = yvf[:, int(yvf.shape[-1] / 2)] * pix_scale[-2]
                    zo = Zvf.T * Zscale
                    C, Z, BA = obj.fit(dtyp, zo, xo, yo,
                                       val=Z.T)  # workaround to get the rotated data from external script (shape can change!)
                    xv, yv = getXYGrid(Z, pix_size, order=2, mask=np.full(Z.shape, True))
                    if retResd:
                        Zfit = obj.get_ellipse(dtyp, xv * pix_scale[-1], C)
                    # Rescale items
                    C[2] = C[2] * 1e3  # theta in mrad
                    C[3] = C[3] * 1e3  # center_offset in mm
                    C[4] = C[4] * 1e3  # tilt in mrad
                    C[5] = C[5] * 1e9  # piston in nm
                    C[6] = C[6] * 1  # rotation in deg
                    Zfit = Zfit / Zscale
                    if BA:
                        Zfit = np.flip(Zfit, axis=1)

            else:
                x0, y0, z0 = calc_center_2d(xvf, yvf, Zvf)
                xf = xf - x0
                Zf = Zf - z0
                err_func, fn_filt = filter_func(elp_params_check, dtyp, ftyp='ellipse', elp_params=elp_params, **kwargs)
                # C, _ = curve_fit(err_func, (fn_filt, xf), Zf, p0=elp_params)
                Cf = np.array(elp_params)[np.array(elp_params_check, dtype=bool)]
                Cnf = np.array(elp_params)[~np.array(elp_params_check, dtype=bool)]
                res = scipy.optimize.least_squares(err_func, x0=Cf, args=(Cnf, xf, Zf, fn_filt), loss='linear',
                                                   xtol=1e-4, ftol=1e-12, gtol=1e-15)
                C = get_C(res.x, Cnf, elp_params_check)
                if retResd:
                    Zfit = fn_filt(xv, *C)
    elif typ == 'legendre':
        A, _, _ = legendre_xy(xf, yf, start_degree=start_degree, degree=degree)
        C, _, _, _ = scipy.linalg.lstsq(A, Zf)
        if retResd:
            _, _, Zfit = zernike_xy(xv, yv, degree=degree, coef=C)
    elif typ == 'zernike':
        A, _, _ = zernike_xy(xf, yf, start_degree=start_degree, degree=degree)
        C, _, _, _ = scipy.linalg.lstsq(A, Zf)
        if retResd:
            _, _, Zfit = zernike_xy(xv, yv, degree=degree, coef=C)
    else:  # polynomial by default
        if np.any(filter_terms_poly):
            degree = -1
        A = matPoly(xf, yf, degree, start_degree, nbVar=2, terms=filter_terms_poly, dtyp=dtyp)
        C, _, _, _ = scipy.linalg.lstsq(A, Zf)  # coefficients C[0] + C[1]*Y + C[2]*X + C[3]*Y2+ C[4]*XY + C[5]*X2
        if retResd:
            Zfit = evalPoly(C, xv, yv, degree, start_degree, nbVar=2, terms=filter_terms_poly, dtyp=dtyp)

    if 'rescale' in more_opts and more_opts['rescale'] and 'rescale_type' in more_opts:
        if more_opts['rescale_type'] == 0:  # Auto scale
            Zfit = Zfit / scale
    if 'center_data' in more_opts and more_opts['center_data']:
        Zfit = Zfit + z0
    resd = Z - Zfit if retResd and (Zfit is not None) else None
    return C, resd, Zfit


def calc_center_2d(xvf, yvf, Zvf):
    xo = xvf[int(xvf.shape[-2] / 2), :]
    yo = yvf[:, int(yvf.shape[-1] / 2)]
    x0 = np.nanmean(xo)
    y0 = np.nanmean(yo)
    ix = np.argmin(np.absolute(xo - x0))
    iy = np.argmin(np.absolute(yo - y0))
    z0 = scipy.interpolate.interp2d(xo[ix - 3:ix + 4], yo[iy - 3:iy + 4], Zvf[iy - 3:iy + 4, ix - 3:ix + 4])(x0, y0)
    return x0, y0, z0


def calc_center_1d(x, Z):
    x0 = np.nanmean(x)
    ix = np.argmin(np.absolute(x - x0))
    z0 = scipy.interpolate.interp1d(x[ix - 3:ix + 4], Z[ix - 3:ix + 4])(x0)
    return x0, z0


def filter_func(F, dtyp, ftyp='poly', **kwargs):
    if ftyp == 'poly':
        if dtyp == 'height':
            return lambda xy, a, b, c, d, e, f, g: F[0] * a + F[1] * b * xy[1] + F[2] * c * xy[0] + F[3] * d * xy[
                1] ** 2 + F[4] * e * xy[0] * xy[1] + F[5] * f * xy[0] ** 2 + F[6] * g * (xy[1] ** 2 + xy[0] ** 2)
        elif dtyp == 'slopes_x':
            return lambda xy, a, b, c, d, e, f, g: F[2] * c + F[4] * e * xy[1] + 2 * F[5] * f * xy[0]
        elif dtyp == 'slopes_y':
            return lambda xy, a, b, c, d, e, f, g: F[1] * b + F[4] * e * xy[0] + 2 * F[3] * d * xy[1]
    elif ftyp == 'ellipse':
        filter_items = F
        elp_params_init = kwargs['elp_params'] if 'elp_params' in kwargs else [0] * 6
        Zscale = kwargs['Zscale'] if 'Zscale' in kwargs else 1
        pix_scale = kwargs['pix_scale'] if 'pix_scale' in kwargs else [1, 1]
        if dtyp == 'height':
            hgt_scale = Zscale
            slp_scale = Zscale / pix_scale[0]
        else:
            hgt_scale = Zscale * pix_scale[0]
            slp_scale = Zscale

        def elp_sfit(x, p, q, theta, center, rotate, piston):
            theta_si = theta * 1e-3
            center_si = center * 1e-3
            rotate_si = rotate * 1e-3
            x = x * pix_scale[0] - center_si
            a = (p + q) / 2
            b = np.sqrt(p * q) * np.sin(theta_si)
            F = (1 / 2) * np.sqrt(p ** 2 + q ** 2 - 2 * p * q * np.cos(np.pi - (2 * theta_si)))
            alpha = np.arcsin(p / (2 * F) * np.sin(np.pi - (2 * theta_si)))
            mu = alpha - theta_si
            x0 = F - q * np.cos(alpha)
            # y0 = -b*np.sqrt(1-(x0/a)**2)

            sx_fit_fn = (np.cos(mu) ** 2) * (b / a) * (
                    (np.cos(mu) * x + x0) /
                    np.sqrt(-1 * (np.cos(mu) ** 2) * (x ** 2) - 2 * np.cos(mu) * x * x0 + (a ** 2) - (x0 ** 2))
            ) + np.cos(mu) * np.sin(-mu) + rotate_si
            sx_fit_fn = sx_fit_fn / slp_scale  # *1e6

            return sx_fit_fn

        def elp_hfit(x, p, q, theta, center, rotate, piston):
            theta_si = theta * 1e-3
            center_si = center * 1e-3
            rotate_si = rotate * 1e-3
            piston_si = piston * 1e-9
            x = x * pix_scale[0] - center_si
            a = (p + q) / 2
            b = np.sqrt(p * q) * np.sin(theta_si)
            F = (1 / 2) * np.sqrt(p ** 2 + q ** 2 - 2 * p * q * np.cos(np.pi - (2 * theta_si)))
            alpha = np.arcsin(p / (2 * F) * np.sin(np.pi - (2 * theta_si)))
            mu = alpha - theta_si
            x0 = F - q * np.cos(alpha)
            y0 = -b * np.sqrt(1 - (x0 / a) ** 2)

            z1 = np.cos(mu) * (-np.sqrt((b ** 2) * (1 - ((x * np.cos(mu) + x0) / a) ** 2)) - y0)
            z2 = np.sin(-mu) * x * np.cos(mu)
            z3 = np.cos(mu) * (y0 + np.sqrt((b ** 2) * (1 - (x0 / a) ** 2)))
            z_fit_fn = z1 + z2 + z3

            # rotate
            if rotate:
                # x        = x * np.cos(rotate_si) + z_fit_fn * np.sin(rotate_si)
                z_fit_fn = x * np.sin(rotate_si) + z_fit_fn * np.cos(rotate_si)
            if piston:
                z_fit_fn = z_fit_fn + piston_si

            z_fit_fn = z_fit_fn / hgt_scale  # *1e9

            return z_fit_fn

        def err_func_cfit(xf, p, q, theta, center, rotate, piston):
            fn_filt, x = xf
            p = p if filter_items[0] == 1 else elp_params_init[0]
            q = q if filter_items[1] == 1 else elp_params_init[1]
            theta = theta if filter_items[2] == 1 else elp_params_init[2]
            center = center if filter_items[3] == 1 else elp_params_init[3]
            rotate = rotate if filter_items[4] == 1 else elp_params_init[4]
            piston = piston if filter_items[5] == 1 else elp_params_init[5]
            return fn_filt(x, p, q, theta, center, rotate, piston)

        def err_func(Cf, Cnf, x, z, fn_filt):
            C = get_C(Cf, Cnf, filter_items)
            print('{}'.format(['{:.7f}'.format(x) for x in C]))
            try:
                err = fn_filt(x, *C) - z
            except Exception as e:
                print(e)
            return err[np.isfinite(err)]

        if dtyp == 'height':
            return err_func, elp_hfit
        elif dtyp == 'slopes_x':
            return err_func, elp_sfit


def get_C(Cf, Cnf, filter_items):
    C = np.array([0.0] * len(filter_items))
    C[np.array(filter_items, dtype=bool)] = Cf
    C[~np.array(filter_items, dtype=bool)] = Cnf
    return C


def matPoly(xf, yf, enDeg, stDeg=0, nbVar=2, terms=[], dtyp='height'):
    """
    Expand IndexMatrix with polynomial terms
    # For filter of terms (bivariate)
    # Format C[0] + C[1]*Y + C[2]*X + C[3]*Y2+ C[4]*XY + C[5]*X2 + C[6]*(Y2+X2) for heights
    # Format C[2] + C[4]*Y + 2*C[5]*X for slopes x
    # Format C[1] + C[4]*X + 2*C[3]*Y for slopes y

    :param xf: X-position array
    :param yf: Y-position array
    :param enDeg: Polynomial degree
    :return: Expanded IndexMatrix
    """
    A = None
    F = terms
    if nbVar == 2:
        if np.any(terms) and enDeg == -1:
            A = np.zeros(shape=(len(xf), 7), dtype=float)
            if dtyp == 'height':
                A[:, 0] = F[0] * 1
                A[:, 1] = F[1] * yf
                A[:, 2] = F[2] * xf
                A[:, 3] = F[3] * yf ** 2
                A[:, 4] = F[4] * xf * yf
                A[:, 5] = F[5] * xf ** 2
                A[:, 6] = F[6] * (xf ** 2 + yf ** 2)
            elif dtyp == 'slopes_x':
                A[:, 2] = F[2] * 1
                A[:, 4] = F[4] * yf
                A[:, 5] = F[5] * 2 * xf
            elif dtyp == 'slopes_y':
                A[:, 1] = F[1] * 1
                A[:, 3] = F[3] * 2 * yf
                A[:, 4] = F[4] * xf
        else:
            cols = nbTermsPoly(stDeg - 1, enDeg)
            A = np.zeros(shape=(len(xf), cols), dtype=float)
            i = 0
            for n in range(stDeg, enDeg + 1):
                for k in range(0, n + 1):
                    # b = np.reshape(xf**(k)*yf**(n-k),(len(xf),1))
                    # A = b if A is None else np.concatenate((A, b),axis=1)
                    A[:, i] = xf ** (k) * yf ** (n - k)
                    i += 1
    elif nbVar == 1:
        if np.any(terms) and enDeg == -1:
            A = np.zeros(shape=(len(xf), 7), dtype=float)
            if dtyp == 'height':
                A[:, 0] = F[0] * 1
                A[:, 2] = F[2] * xf
                A[:, 5] = F[5] * xf ** 2
            elif dtyp == 'slopes_x':
                A[:, 2] = F[2] * 1
                A[:, 5] = F[5] * 2 * xf
        else:
            for n in range(stDeg, enDeg + 1):
                b = np.reshape(xf ** (n), (len(xf), 1))
                A = np.ones((len(xf), 1), dtype='float32') if n == 0 else np.concatenate((A, b), axis=1)
    return A


def evalPoly(C, xv, yv, enDeg, stDeg=0, nbVar=2, terms=[], dtyp='height'):
    """
    Evaluate 2D polynomial

    :param C: Coefficients of polynomial
    :param xv: X-position array
    :param yv: Y-position array
    :param degree: Polynomial degree
    :return: Evaluated data
    """
    Zfit = None
    F = terms
    i = 0
    if isinstance(C, np.ndarray) and C.ndim > 1:
        C = np.rollaxis(C, -1)
        if nbVar == 2:
            C = C.reshape((*C.shape, 1, 1))
        elif nbVar == 1:
            C = C.reshape((*C.shape, 1))
    if nbVar == 2:
        if np.any(terms) and enDeg == -1:
            if dtyp == 'height':
                Zfit = F[0] * C[0] * 1 + F[1] * C[1] * yv + F[2] * C[2] * xv + F[3] * C[3] * yv ** 2 + F[4] * C[
                    4] * xv * yv + F[5] * C[5] * xv ** 2 + F[6] * C[6] * (xv ** 2 + yv ** 2)
            elif dtyp == 'slopes_x':
                Zfit = F[2] * C[2] * 1 + F[4] * C[4] * yv + F[5] * C[5] * 2 * xv
            elif dtyp == 'slopes_y':
                Zfit = F[1] * C[1] * 1 + F[3] * C[3] * 2 * yv + F[4] * C[4] * xv
        else:
            for n in range(stDeg, enDeg + 1):
                for k in range(0, n + 1):
                    val = C[i] * xv ** (k) * yv ** (n - k)
                    Zfit = val if Zfit is None else Zfit + val
                    i += 1
    elif nbVar == 1:
        if np.any(terms) and enDeg == -1:
            if dtyp == 'height':
                Zfit = F[0] * C[0] * 1 + F[2] * C[2] * xv + F[5] * C[5] * xv ** 2
            elif dtyp == 'slopes_x':
                Zfit = F[2] * C[2] * 1 + F[5] * C[5] * 2 * xv
        else:
            for n in range(stDeg, enDeg + 1):
                Zfit = C[i] if n == 0 else Zfit + C[i] * xv ** (n)
                i += 1
    return Zfit


def getXYGrid(oarr, pix_size, order=3, mask=None):
    """
    Get XY grid of positions

    :param oarr: Input data
    :param pix_size: Pixel size
    :param order: Dimensions of input data
    :param mask: Input mask
    :return:
    """
    order = oarr.ndim
    pix_size = getPixSz2D(pix_size)
    if order == 3:
        patch_count, ny, nx = oarr.shape
        mask = mask if np.any(mask) else ~np.isnan(oarr)
        xx = np.cumsum(np.ones(mask.shape), axis=-1) * mask * 1.0
        yy = np.cumsum(np.ones(mask.shape), axis=-2) * mask * 1.0
        xx[xx == 0] = np.nan
        yy[yy == 0] = np.nan
        xx = pix_size[0] * (xx - np.nanmedian(xx, axis=[-1, -2]).reshape(patch_count, 1, 1))
        yy = pix_size[1] * (yy - np.nanmedian(yy, axis=[-1, -2]).reshape(patch_count, 1, 1))
        # xx                          = pix_size[0]*(xx - np.nanmean(xx,axis=-1).reshape(patch_count,ny,1))
        # yy                          = pix_size[1]*(yy - np.nanmean(yy,axis=-2).reshape(patch_count,1,nx))
        return xx, yy
    elif order == 2:
        ny, nx = oarr.shape
        mask = mask if np.any(mask) else ~np.isnan(oarr)
        xx = np.cumsum(np.ones(mask.shape), axis=-1) * mask * 1.0
        yy = np.cumsum(np.ones(mask.shape), axis=-2) * mask * 1.0
        xx[xx == 0] = np.nan
        yy[yy == 0] = np.nan
        xx = pix_size[0] * (xx - np.nanmedian(xx.ravel()))
        yy = pix_size[1] * (yy - np.nanmedian(yy.ravel()))
        # xx                          = pix_size[0]*(xx - np.nanmean(xx,axis=-1).reshape(ny,1))
        # yy                          = pix_size[1]*(yy - np.nanmean(yy,axis=-2).reshape(1,nx))
        return xx, yy


def getPixSz2D(pix_size):
    """
    Check if input is array, tuple or list and return pixel size in x and y

    :param pix_size: Pixel size in differnt formats
    :return: Pixel size XY array
    """
    if type(pix_size) in [list, tuple, np.ndarray]:
        # if not any(pix_size):
        if len(pix_size) == 0:
            return [1, 1]
        if len(pix_size) != 2:
            return [pix_size[0], pix_size[0]]
        else:
            return pix_size
    elif type(pix_size) is str:
        return [float(pix_size), float(pix_size)]
    else:
        return [pix_size, pix_size]
