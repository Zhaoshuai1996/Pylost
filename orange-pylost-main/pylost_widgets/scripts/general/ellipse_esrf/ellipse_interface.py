# coding=utf-8
from pylost_widgets.util.base.EllipseBase import EllipseBase

from PyLOSt.data_in.esrf.ellipse import Ellipse
from PyLOSt.data_in.esrf.generic import Profile, Surface
from PyLOSt.data_in.esrf import units as u

import numpy as np


class EllipseESRFLab(EllipseBase):
    units = {
        'coords': u.m, 'values': u.m,
        'height': u.m, 'angle': u.rad,
        'length': u.m, 'radius': u.m,
    }

    def __init__(self, ellipse_params, checked_params=[0, 1, 1, 0, 1, 0]):
        super(EllipseESRFLab, self).__init__(ellipse_params, checked_params)

    def fit(self, dtype, data, x, y=None, val=None):
        self.units['values'] = u.m if dtype == 'height' else u.rad
        p, q, theta = self.ellipse_params[:3]
        params = self.ellipse_params
        check = np.array(self.checked_params, dtype=bool)
        check[3] = False  # remove offset from params
        rotation = check[4]
        optimization = list(np.array(['p', 'q', 'theta', 'offset', 'piston', 'tilt'])[check])
        data_obj = None
        if data.ndim == 2:
            data_obj = Surface((x, y), data, self.units, source='MetrologyData (orange-pylost)')
        elif data.ndim == 1:
            data_obj = Profile(x, data, self.units, source='MetrologyData (orange-pylost)')
        if data_obj is not None:
            data_obj.set_ellipse(p, q, theta)
            optimized = data_obj.ellipse.optimize(optimization, rotation=rotation)
            params = np.array(
                [optimized.p, optimized.q, optimized.theta, optimized.offset, optimized.tilt, optimized.piston,
                 optimized.rotation])
        data_obj.values = val
        data_obj.units_to_SI()
        if data.ndim == 1:
            return params, data_obj.values, data_obj.ellipse.reversed
        if rotation:
            data_obj.rotate(params[-1])
        return params, data_obj.values.T, data_obj.ellipse.reversed

    def get_ellipse(self, dtype, x, ellipse_params):
        p, q, theta, offset, tilt, piston, rotation = ellipse_params
        return Ellipse.from_parameters(x, dtype, p, q, theta, piston, tilt)
