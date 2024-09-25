# -*- coding: utf-8 -*-
"""
The 'ColorScale' class by Francois Perrin (ESRF) for better visualization of colors for 2d height/slope errors.
It implements color scaling based on peak to valley / standard deviation / mean absolute deviation etc.
"""
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap as lsc
import numpy as np
from pylost_widgets.util import math


class ColorScale:
    """
    Class to manage colormap scaling adjusted by data rms or pv or mad etc. parameters
    """

    def __init__(self, surface, cmap='turbo', name='custom', base='ra', zfac=2.0, **zparams):
        """
        Create and custom linear segmented colormap based on statistical or fixed parameters.

        :param surface: 2d surface height/slope data
        :type surface: ndarray
        :param cmap: matplotlib colormap name
        :type cmap: str
        :param name: name for the new colormap
        :type name: str
        :param base: scaling method ('ra', 'rq', 'pv', 'fixed')
        :type name: str
        :param zfac: stretching factor
        :type zfac: float
        :param zparams: Additional parameters for fixed scaling (zmin, zmax, z_lo, z_hi)
        :type zparams: float
        """
        self.lut = cm.get_cmap(cmap)(np.linspace(0, 1, 256))
        self.surface = surface.ravel()
        self.cmap_prms = {'cmap': cmap, 'base': base}
        self.name = name
        zparams['zfac'] = zfac
        self.cmap, new_params = self._update(**zparams)
        self.cmap_prms.update(new_params)

    def __str__(self):
        string = 'ColorScale:'
        for key, value in self.cmap_prms.items():
            fmt = '0.3f'
            if isinstance(value, str):
                fmt = 's'
            string = string + f' {key}={value:{fmt}}'
        return string

    def _update(self, base='ra', **zparams):
        """
        Update colormap data.
        """
        if zparams.get('zfac', -1) < 0.001:  # force default
            zparams.pop('zfac', 2.0)
        new_params = getattr(self, f'_{base}_based')(**zparams)
        new_params = dict(zip(['zfac', 'zmin', 'zmax', 'z_lo', 'z_hi'], new_params))
        return self._calc_lsc(self.lut, self.name, math.nanmean(self.surface), **new_params), new_params

    def _ra_based(self, zfac=2.0, **kwargs):
        """
        Stretch color scaling within intermediate values at "mean absolute deviation (mad)" from median.
        """
        ra = math.mad(self.surface)
        rq = math.nanstd(self.surface)
        data = math.mask_outliers(self.surface, 3.0*rq)
        median = math.nanmedian(data)
        z_lo = -zfac * ra + median
        z_hi = zfac * ra + median
        return zfac, math.nanmin(data), math.nanmax(data), z_lo, z_hi

    def _rq_based(self, zfac=3.0, **kwargs):
        """
        Stretch color scaling within intermediate values at "standard deviation" from median.
        """
        rq = math.nanstd(self.surface)
        data = math.mask_outliers(self.surface, 6.0*rq)
        median = math.nanmedian(data)
        z_lo = -zfac * rq + median
        z_hi = zfac * rq + median
        return zfac, math.nanmin(data), math.nanmax(data), z_lo, z_hi

    def _pv_based(self, zfac=12.0, **kwargs):
        """
        Stretch color scaling within intermediate values at "standard deviation / peak to valley" from median.
        """
        rq = math.nanstd(self.surface)
        rv = math.nanmin(self.surface)
        rp = math.nanmax(self.surface)
        median = math.nanmedian(self.surface)
        ratio = zfac * rq / (rp - rv)
        z_lo = -ratio * rq + median
        z_hi = ratio * rq + median
        return zfac, rv, rp, z_lo, z_hi

    def _fixed_based(self, zfac=3.0, **kwargs):
        """
        Stretch color based on fixed values
        """
        if len(kwargs) == 4:
            return [zfac, ] + list(kwargs.values())
        zmin = np.negative(zfac)
        zmax = zfac
        median = math.nanmedian(self.surface)
        z_lo = 0.40 * zmin + median
        z_hi = np.negative(z_lo)
        return zfac, zmin, zmax, z_lo, z_hi

    @staticmethod
    def _calc_lsc(lut, name, mean, zmin, zmax, z_lo, z_hi, **kwargs):
        """
        Calculate new linear segmented colormap by dividing linear scale into 4 regions which are redivided into
        different number of points, e.g.within [data-min, rq-min, rq-max, data-max]
        """
        num = 240
        z = (np.linspace(zmin, zmax, num=num, endpoint=True))
        z_lo1 = np.linspace(zmin, z_lo, num=15, endpoint=False)
        z_lo2 = np.linspace(z_lo, mean, num=105, endpoint=False)
        z_lo0 = np.append(z_lo1, z_lo2)
        z_hi1 = np.linspace(mean, z_hi, num=85, endpoint=False)
        z_hi2 = np.linspace(z_hi, zmax, num=35, endpoint=True)
        z_hi0 = np.append(z_hi1, z_hi2)
        new_z = np.append(z_lo0, z_hi0)
        new_z = np.interp(z, new_z, z)
        idx = 0
        blue = lut[idx:idx + num, 0]
        green = lut[idx:idx + num, 1]
        red = lut[idx:idx + num, 2]
        new_cmap = np.array([np.interp(new_z, z, blue),
                             np.interp(new_z, z, green),
                             np.interp(new_z, z, red)]).T
        return lsc.from_list(name, new_cmap, num)
