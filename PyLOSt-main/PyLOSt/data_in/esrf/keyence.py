# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:05:38 2020

"""

# pylint: disable=C0103, C0115, C0116
# pylint: disable=R0902, R0903, R0904


import os
import datetime as dt
import numpy as np
from pathlib import Path

try:
    import swift.units as u
    from esrf.data.generic import ESRFOpticsLabData, Surface
    import esrf.data.vk4extract as vk4extract
except ImportError:
    from . import units as u
    from .generic import ESRFOpticsLabData, Surface
    from . import vk4extract as vk4extract


class KeyenceData(ESRFOpticsLabData, Surface):
    '''Keyence VK4 data class.'''
    method = 'Confocal microscope'
    instrument = "VK-X1100"

    def __init__(self):
        super().__init__()

        self.header_format = None
        self.header_size = None
        self.note = None

        self.datetime = None

        self._raw_shape = None
        self.intensity = None
        self.phase = None

        self.units = {
            'coords': u.um, 'values': u.um,
            'height': u.nm, 'angle': u.urad,
            'length': u.mm, 'radius': u.km,
            'pixel': u.um,
        }

    def __str__(self):
        return 'Keyence surface map'

    # ----overriding----
    def readfile(self, path, source=None):  # pylint: disable=unused-argument, R0914
        with open(path, 'rb') as input_file:
            self.datetime = dt.datetime.fromtimestamp(os.path.getmtime(path))

            if isinstance(path, str):
                path = Path(path)
            path = path
            self.source = path.name

            print(f'opening \'{self.source}\'...')

            offsets = vk4extract.extract_offsets(input_file)
            self.header['offsets'] = offsets
            meas_cond = vk4extract.extract_measurement_conditions(offsets, input_file)
            self.header['meas_cond'] = meas_cond
            height_arr = vk4extract.extract_img_data(offsets, 'height', input_file)
            int_arr = vk4extract.extract_img_data(offsets, 'light', input_file)

            h, w = height_arr['height'], height_arr['width']
            spixX, spixY, spixZ = meas_cond['x_length_per_pixel'], meas_cond['y_length_per_pixel'], meas_cond[
                'z_length_per_digit']
            arr = np.resize(height_arr['data'], (h, w)).T * spixZ * 1e-6
            intensity = np.resize(int_arr['data'], (h, w)).T
            fVx, fVy = w * spixX * 1e-6, h * spixY * 1e-6
            fieldView = [-0.5 * fVx, 0.5 * fVx, -0.5 * fVy, 0.5 * fVy]

            self.header['lateral_res'] = np.float64(spixX * 1e-6)
            self._raw_shape = arr.shape
            x = np.linspace(0, self._raw_shape[0] * self.header['lateral_res'],
                            num=self._raw_shape[0], endpoint=False, dtype=np.float64)
            y = np.linspace(0, self._raw_shape[1] * self.header['lateral_res'],
                            num=self._raw_shape[1], endpoint=False, dtype=np.float64)
            self.initial = Surface((x, y), arr, self.units, self.source)

        return self
