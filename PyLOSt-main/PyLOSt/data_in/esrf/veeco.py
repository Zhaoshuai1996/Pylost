# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:05:38 2020

https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/

http://research.endlessfernweh.com/wp-content/uploads/week4_fitting.pdf

http://logical.ai/conic/org/fitting.html

https://core.ac.uk/download/pdf/82101893.pdf

https://www.sciencedirect.com/science/article/pii/S0024379503007535
"""

# pylint: disable=C0103, C0115, C0116
# pylint: disable=R0902, R0903, R0904

import datetime
import numpy as np
from pathlib import Path

try:
    import swift.units as u
    from esrf.data.generic import ESRFOpticsLabData, Surface
except ImportError:
    from . import units as u
    from .generic import ESRFOpticsLabData, Surface

BPF = 1e+38
BLCK_SIZE = 24


class OpdData(ESRFOpticsLabData, Surface):
    '''Vision32 data class.'''
    method = 'White Light Interferometry'
    instrument = "WYKO NT 9300 (Veeco)"

    def __init__(self):
        super().__init__()

        self.header_size = None
        self.note = None

        self._raw_shape = None

        self.units = {
            'coords': u.mm, 'values': u.nm,
            'height': u.nm, 'angle': u.urad,
            'length': u.mm, 'radius': u.km,
            'pixel': u.mm,
        }

    def __str__(self):
        return 'Veeco 2D surface map'

    # ----overriding----
    def readfile(self, path, source=None):  # pylint: disable=unused-argument
        with open(path, 'rb') as opd_file:

            if isinstance(path, str):
                path = Path(path)
            path = path
            self.source = path.name

            block_ppos = 0
            directory = self.opd_block(opd_file, block_ppos)
            if 'Directory' not in directory.name:
                print('Unable to read Directory entry, aborting...')
                return self
            self.header_size = directory.len
            num_block_max = int(self.header_size / (BLCK_SIZE + 2))

            blocks = []
            for b in range(1, num_block_max + 1):  # pylint: disable=unused-variable
                block_ppos += BLCK_SIZE
                block = self.opd_block(opd_file, block_ppos)
                if block.signature != 0:
                    blocks.append(block)

            opd_file.seek(self.header_size + 2)
            for block in blocks:
                block.decode_value(opd_file)
                if block.name in ('RAW_DATA', 'Raw'):
                    rawdata = np.where(block.value < BPF, block.value, np.nan)
                else:
                    self.header[block.name] = block.value

            # stage positions: inches to mm
            if 'StageX' in self.header:
                self.header['StageX'] = self.header['StageX'] * 25.4
            if 'StageY' in self.header:
                self.header['StageY'] = self.header['StageY'] * 25.4

            self.header['Pixel_size'] = np.float64(self.header['Pixel_size'])
            self.header['Wavelength'] = np.float64(self.header['Wavelength'])

            # nanometers
            rawdata = rawdata * self.header['Wavelength']
            rawdata -= np.nanmean(rawdata)
            self._raw_shape = rawdata.shape
            if self._raw_shape[1] / self._raw_shape[0] != 0.75:  # remove nan lines/columns in Vision32 stitching
                rawdata = rawdata[~(np.isnan(rawdata).all(axis=1)), :]
                rawdata = rawdata[:, ~(np.isnan(rawdata).all(axis=0))]
            x = np.linspace(0, self._raw_shape[0] * self.header['Pixel_size'],
                            num=self._raw_shape[0], endpoint=False, dtype=np.float64)
            y = np.linspace(0, self._raw_shape[1] * self.header['Pixel_size'],
                            num=self._raw_shape[1], endpoint=False, dtype=np.float64)
            self.initial = Surface((x, y), rawdata, self.units, self.source)
        return self

    # ----properties----
    @property
    def datetime(self):
        date = self.header['Date']
        time = self.header['Time']
        return datetime.datetime.strptime(f'{date} {time}', '%m/%d/%Y %H:%M:%S')

    @property
    def title(self):
        return self.header.get('Title', '')

    @property
    def mode(self):
        return self.header.get('MeasMode', '')

    @property
    def objective(self):
        return self.header.get('ObjectiveLabel', self.magnification)

    @property
    def magnification(self):
        mag = self.header.get('Magnification', 0)
        return f'{mag:.1f} X'

    @property
    def fov(self):
        default = '1 X'
        return self.header.get('FOVLabel', default)

    @property
    def stage_x(self):
        try:
            return self.header['StageX']  # mm
        except KeyError:
            return 0.0

    @property
    def stage_y(self):
        try:
            return self.header['StageY']  # mm
        except KeyError:
            return 0.0

    @property
    def average(self):
        return self.header['Averages']

    @property
    def wavelength(self):
        return self.header['Wavelength']  # nm

    # @property
    # def histogram(self):
    #     mask = self.masking()
    #     return np.histogram(self.values[mask], bins=512, density=False)
    #     # plt.plot(histo[1][:-1], histo[0])

    class opd_block():
        def __init__(self, file, ppos):
            file.seek(ppos)
            block = file.read(BLCK_SIZE + 2)
            self.signature = np.frombuffer(block[0:2], dtype=np.short)[0]
            if self.signature == 0:
                return
            try:
                self.name = block[2:18].decode().rstrip('\x00')
            except UnicodeDecodeError:
                self.name = block[2:18].decode('latin1').rstrip('\x00')
            self.type = np.frombuffer(block[18:20], dtype=np.short)[0]
            self.len = np.frombuffer(block[20:24], dtype=np.int32)[0]
            self.attr = np.frombuffer(block[24:26], dtype=np.ushort)[0]
            self.value = None

        def decode_value(self, file):
            buf = file.read(self.len)
            if self.type == 3:  # array of data
                rows, cols, elsize = np.frombuffer(buf[0:6], dtype=np.ushort)[0:3]
                dtype = np.single
                if elsize == 2:
                    dtype = np.short
                self.value = np.resize(np.frombuffer(buf[6:], dtype=dtype),
                                       (rows, cols))
            elif self.type == 5:  # string
                self.value = buf.decode().strip('\x00')
            elif self.type == 6:  # short
                self.value = np.frombuffer(buf, dtype=np.short)[0]
            elif self.type == 7:  # float
                if len(buf) == 4:
                    self.value = np.float64(np.frombuffer(buf, dtype=np.single)[0])
                if len(buf) == 2:
                    self.value = np.float64(np.frombuffer(buf, dtype=np.half)[0])
            elif self.type == 8:  # double
                self.value = np.float64(np.frombuffer(buf, dtype=np.double)[0])
            elif self.type == 12:  # long
                self.value = np.frombuffer(buf, dtype=np.int32)[0]
