# -*- coding: utf-8 -*-
"""

    ellipse class

"""

# pylint: disable=C0103, C0115, C0116
# pylint: disable=R0902, R0903, R0913, R0914

from time import perf_counter_ns

import numpy as np

from scipy.optimize.minpack import leastsq
# from scipy.optimize import least_squares

from scipy.optimize import fminbound

try:
    from esrf.data.generic import Profile
except ImportError:
    from .generic import Profile


class Ellipse():
    """p and q in meter, theta in mrad"""

    def __init__(self, measurement, p, q, theta):
        self.p = p
        self.q = q
        self.theta = theta * 1e-3
        self.piston = 0.0
        self.tilt = 0.0
        self.offset = 0.0
        self.rotation = 0.0
        self.reversed = False
        self.optimizations = dict()
        self.optimized = None
        self.SI = measurement.duplicate()
        self.SI.units_to_SI()
        self.SI.center_coordinates()
        edge = int(len(self.SI.x) / 8)
        lim = int(len(self.SI.x) / 64)
        if 'Profile' in str(type(self.SI)):
            if 'slope' in self.SI.kind.lower():
                self.SI.level_data()
                polyfit = self.SI.duplicate()
            else:
                self.SI.polynomial_removal()
                self.SI.min_removal()
                self.reversed = np.nanmean(self.SI.values[lim:edge]) > np.nanmean(self.SI.values[-edge:-lim])
        elif 'Surface' in str(type(self.SI)):
            if 'height' in self.SI.kind.lower():
                self.SI.plane_removal(reload=False, auto_units=False)
                self.SI.min_removal()
                self.reversed = np.nanmean(self.SI.values[lim:edge, :]) > np.nanmean(self.SI.values[-edge:-lim, :])
            else:
                self.SI.level_data()
                polyfit = Profile(self.SI.x, np.nanmean(self.SI.values, axis=1), self.SI.units)
        else:
            raise ValueError('not a valid pyopticslab data class')
        if 'slope' in self.SI.kind.lower():
            poly = polyfit.polynomial_removal(6)  # !!! assuming noisy data
            curv_left = np.nanmean(np.gradient(poly[lim:edge]))
            curv_right = np.nanmean(np.gradient(poly[-edge:-lim]))
            self.reversed = curv_left > curv_right
        if self.reversed:
            self.SI.flip()
        self.SI.initial = self.SI.duplicate()
        self.theoretical, self.residues = self.from_class()

    def __str__(self):
        # return ','.join(self.optimization)
        fmt = '.7f'
        return f'p:{self.p:{fmt}} q:{self.q:{fmt}} theta:{self.theta * 1e3:{fmt}}'

    def from_class(self):
        new_ellipse = self.SI.duplicate()
        if self.SI.values.ndim == 1:
            x = new_ellipse.coords
        elif self.SI.values.ndim == 2:
            x, _ = new_ellipse.meshgrid
        else:
            raise ValueError('not a valid pyopticslab data class')
        new_ellipse.values = Ellipse.from_parameters(x, new_ellipse.kind,
                                                     self.p, self.q, self.theta,
                                                     self.piston, self.tilt)
        residues = self.SI.duplicate()
        residues.values = self.SI.values - new_ellipse.values
        if residues.values.ndim == 2:
            residues.plane_removal()
        return new_ellipse, residues

    @staticmethod
    def from_parameters(x, kind, p, q, theta, piston=0.0, tilt=0.0):
        """theoretical ellipse (meter, radian). Returns slopes or heights"""
        a = (p + q) / 2.0  # semi-major axis
        b = np.sqrt(p * q) * np.sin(theta)  # semi-minor axis
        F = np.sqrt(a ** 2 - b ** 2)  # linear eccentricity
        alpha = np.arcsin(p * np.sin(2.0 * theta) / (2.0 * F))
        mu = alpha - theta
        # ecc = np.sqrt(a**2-np.sqrt(p*q)**2)/a # eccentricity with b=np.sqrt(p*q)
        # mu2 = theta * ecc
        x0 = F - q * np.cos(alpha)
        # y0 = -b*np.sqrt(1 - (x0/a)**2)
        if 'slope' in kind.lower():
            s1n = (b / a) * (np.cos(mu)) ** 2 * (x0 + x * np.cos(mu))
            s1d = a ** 2 - x0 ** 2 - (x * np.cos(mu)) ** 2 - 2.0 * x * x0 * np.cos(mu)
            s1 = s1n / np.sqrt(s1d)
            s2 = np.cos(mu) * -np.sin(mu)
            sx = s1 + s2
            return sx + tilt
        # heights
        x = x * np.cos(tilt)
        z1 = x * np.cos(mu) * -np.sin(mu)
        z2 = np.sqrt(b ** 2 * (1.0 - (x0 / a) ** 2))
        z3 = np.sqrt(b ** 2 * (1.0 - ((x0 + x * np.cos(mu)) / a) ** 2))
        hx = z1 + np.cos(mu) * (z2 - z3)
        hx = x * np.sin(tilt) + hx * np.cos(tilt)
        return hx + piston

    def optimize(self, optimization=('q', 'theta'), offset=False, rotation=False):
        rotation = rotation and 'height' in self.SI.kind and (self.SI.values.ndim == 2)
        self.optimized = Optimization(self, optimization, offset, rotation)
        if rotation:
            self.optimized.SI.values[0, :] = np.nan
            self.optimized.SI.values[-1, :] = np.nan
        self.optimized.theoretical, self.optimized.residues = self.optimized.from_class()
        title = f'\n{self.SI.source}'
        if self.reversed:
            self.optimized.theoretical.flip()
            self.optimized.residues.flip()
            title = title + ' (BA scan?)'
        self.residues = self.optimized.residues
        self.p = self.optimized.p
        self.q = self.optimized.q
        self.theta = self.optimized.theta
        self.optimizations[self.optimized] = self.optimized
        title = title + ':'
        print(title)
        print(f'    {self.optimized.descr}\n')
        return self.optimized


class Optimization(Ellipse):
    """ optimization: (tuple/list) including one or more following key:
                'p', 'q', 'theta', 'piston', 'tilt'
    """

    def __init__(self, ellipse, optimization=('q', 'theta'), offset=False, rotation=False):
        super().__init__(ellipse.SI, ellipse.p, ellipse.q, ellipse.theta * 1e3)
        self.descr = ''
        if isinstance(optimization, tuple):
            optimization = list(optimization)
        # force piston & tilt optimization
        if 'piston' not in optimization:
            optimization = optimization + ['piston', ]
        if 'tilt' not in optimization:
            optimization = optimization + ['tilt', ]

        if 'offset' in optimization:
            optimization.remove('offset')

        self.optimization = optimization
        self.opt_prms = None
        self.theta = self.theta
        self.piston = 0.0
        self.tilt = 0.0
        self.offset = 0.0
        self.rotation = 0.0
        self.id = self._optimize(offset, rotation, ellipse.reversed)

    def __str__(self):
        return self.id

    # ----properties----
    @property
    def x(self):
        if self.SI.values.ndim == 1:
            x = self.SI.coords
        elif self.SI.values.ndim == 2:
            x, _ = self.SI.meshgrid
        else:
            raise ValueError('not a valid pyopticslab data class')
        return x[self.mask]

    @property
    def z(self):
        return self.SI.values[self.mask]

    @property
    def mask(self):
        return ~np.isnan(self.SI.values)

    def get_opt_params(self, params):
        self.opt_prms['x'] = self.x
        for i, key in enumerate(self.optimization):
            self.opt_prms[key] = params[0][i]
        return self.opt_prms

    class Wrap():  # pylint: disable=unused-argument
        @staticmethod
        def ellipse_removal(func):  # pylint: disable=unused-argument
            def _wrapping(data, *params):
                # print(params)
                return data.z - Ellipse.from_parameters(**data.get_opt_params(params))

            return _wrapping

    @Wrap.ellipse_removal
    def _wrapped_loop(parameters):  # pylint: disable=no-self-argument
        return parameters

    def _lstsq(self, p0):  # TODO: check performance
        # binf = []
        # bsup = []
        # for p in p0:
        #     ppct = abs(p*0.05)
        #     if abs(p) < 1e-9:
        #         ppct = 1e-3
        #     binf.append(p-ppct)
        #     bsup.append(p+ppct)
        # bounds = (binf, bsup)
        # print(f'parameters bounds={bounds}')
        # return least_squares(self._wrapped_loop, p0, bounds=bounds,
        #                 xtol=1e-4, ftol=1e-12, gtol=1e-15,
        #                 verbose=1)
        return leastsq(self._wrapped_loop, p0,
                       xtol=1e-4, ftol=1e-12, gtol=1e-15,
                       epsfcn=0.0001, factor=10,
                       maxfev=50,
                       full_output=True)

    def _offset(self, offset):  # , p0):
        self.opt_prms['x'] = self.x - offset
        res = self.z - Ellipse.from_parameters(**self.opt_prms)
        ret = np.nanmax(res) - np.nanmin(res)
        # ret = np.nanstd(res)
        # print(offset, ret)
        return ret

    def _rotate(self, deg):
        self.SI.reload()
        self.SI.rotate(deg)
        self.opt_prms['x'] = self.x
        res = self.z - Ellipse.from_parameters(**self.opt_prms)
        # ret = np.nanmax(res) - np.nanmin(res)
        ret = np.nanstd(res)
        # print(deg, ret*1e6)
        return ret * 1e6

    def _optimize(self, offset=False, rotation=False, reverse=False):
        kind = self.SI.kind
        self.opt_prms = {'x': self.x,
                         'kind': kind,
                         'p': self.p,
                         'q': self.q,
                         'theta': self.theta,
                         'piston': 0.0,
                         'tilt': 0.0,
                         }
        p0 = []
        for key in self.optimization:
            p0.append(getattr(self, key))

        tic = perf_counter_ns()
        if offset:
            popt = fminbound(self._offset, -0.5, 0.5, xtol=1e-4,
                             full_output=True, maxfun=100, disp=2)
            nfev = popt[3]
            self.offset = popt[0]
        if reverse:
            self.offset = -self.offset
        if rotation:
            for i in range(0, 2):
                popt = self._lstsq(p0)
                popt = fminbound(self._rotate, -0.5, 0.5, xtol=1e-4,
                                 full_output=True, maxfun=100, disp=2)
                self.rotation = popt[0]
                nfev = popt[3]
            self.SI.reload()
            self.SI.rotate(self.rotation)
        popt = self._lstsq(p0)
        toc = perf_counter_ns()
        # print(popt)
        try:
            nfev = popt.nfev
        except:
            nfev = popt[2]['nfev']

        self.p = self.opt_prms['p']
        self.q = self.opt_prms['q']
        self.theta = self.opt_prms['theta']
        self.piston = self.opt_prms['piston']
        self.tilt = self.opt_prms['tilt']

        [self.opt_prms.pop(key) for key in ['x', 'kind']]  # pylint: disable=W0106
        string = ', '.join(self.optimization)
        if offset:
            string = string + ', offset'
        if rotation:
            string = string + ', rotation'
        del popt
        self.descr = f'({string}) optimization based on {kind} (N={nfev}) ({(toc - tic) * 1e-6:.0f}ms):\n        '
        fmt = '.7f'
        # for key, val in self.opt_prms.items():
        #     self.descr = self.descr + f' {key}:{val:{fmt}}'
        for key in ('p', 'q', 'theta'):
            self.descr = self.descr + f' {key}:{self.opt_prms[key]:{fmt}}'
        if 'piston' in self.optimization:
            key = 'piston'
            self.descr = self.descr + f' piston:{self.opt_prms[key]:.2e}'
        if 'tilt' in self.optimization:
            key = 'tilt'
            self.descr = self.descr + f' tilt:{self.opt_prms[key]:.2e}'
        if offset:
            self.descr = self.descr + f' offset:{self.offset:.3f}'
        if rotation:
            self.descr = self.descr + f' rotation:{self.rotation:.3f}deg'
        return kind + ',' + '_'.join(self.optimization)
