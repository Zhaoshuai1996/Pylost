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
import h5py

try:
    import swift.units as u
    from esrf.data.generic import ESRFOpticsLabData, Surface
except ImportError:
    from . import units as u
    from .generic import ESRFOpticsLabData, Surface


class MxData(ESRFOpticsLabData, Surface):
    '''MX data class.'''
    method = 'Fizeau Interferometer'
    instrument = "Zygo"

    def __init__(self):
        super().__init__()

        self.datetime = None

        self._raw_shape = None
        self.intensity = None
        self.phase = None

        self.units = {
            'coords': u.m, 'values': u.m,
            'height': u.nm, 'angle': u.urad,
            'length': u.mm, 'radius': u.km,
            'pixel': u.mm,
        }

        self.attributes = None
        # self.measurement_surface = None
        # self.measurement_intensity = None
        # self.lateral_resolutions = None

    class Struct:
        def __init__(self, file_obj):
            self.metadata = self._metadata(file_obj)
            self.attributes = self._measurement_attributes(file_obj)
            self.surface, self.surface_attrs = self._surface(file_obj)
            # self.intensity, self.intensity_attrs = self._intensity(file_obj)

        @staticmethod
        def _metadata(datx_struct):
            source = datx_struct['MetaData']['Source'].astype(str)
            link = datx_struct['MetaData']['Link'].astype(str)
            dest = datx_struct['MetaData']['Destination'].astype(str)
            meta = {}
            for s, l, d in zip(source, link, dest):
                if meta.get(s, None) is None:
                    meta[s] = {}
                path = d.split('/')
                if len(path) > 1:
                    d = path[1:]
                meta[s][l] = d
            return meta

        @staticmethod
        def _find_data(parent, path_parts):
            child = parent[path_parts[0]]
            if len(path_parts) < 2:
                return child
            return MxData.Struct._find_data(child, path_parts[1:])

        @property
        def timestamp(self):
            stamp = self.attributes['Data']['Time Stamp']['time']
            return dt.datetime.utcfromtimestamp(stamp)

        @property
        def _measurement_guid(self):
            return self.metadata['Root']['Measurement']

        def _measurement_attributes(self, datx_struct):
            attr_path = self.metadata[self._measurement_guid]['Attributes']
            attrs = self._find_data(datx_struct, attr_path).attrs
            return self._measurement_attributes_to_dict(attrs)

        @property
        def _surface_guid(self):
            return self.metadata[self._measurement_guid].get('Surface', None)

        def _surface(self, datx_struct):
            data_path = self.metadata[self._surface_guid]['Path']
            dataset = self._find_data(datx_struct, data_path)
            if dataset.compression == 'szip':
                raise Exception('File uses szip compression, resave with a earlier version of MX (>=5.25.0.1).')
            data = np.zeros(dataset.shape)
            dataset.read_direct(data)
            return data, self._data_attributes_to_dict(dataset.attrs)

        @property
        def _intensity_guid(self):
            return self.metadata[self._measurement_guid].get('Intensity', None)

        def _intensity(self, datx_struct):
            data_path = self.metadata[self._intensity_guid]['Path']
            dataset = self._find_data(datx_struct, data_path)
            data = np.zeros(dataset.shape)
            dataset.read_direct(data)
            return data, self._data_attributes_to_dict(dataset.attrs)

        @staticmethod
        def _measurement_attributes_to_dict(attrs):
            dico = {}
            bag_list = [p.split('.')[0] for p in attrs['Property Bag List'][0].strip().split('\n')]
            for k in bag_list:
                key_name = k.split(' ')[0]
                dico[key_name] = {}
            for key, val in attrs.items():
                attr_path = key.split('.')
                if attr_path[0] in bag_list:
                    key_name = attr_path[0].split(' ')[0]
                    attr_name = attr_path[-1]  # .replace(' ', '_')
                    if isinstance(val[0], np.void):
                        # continue
                        dico[key_name][attr_name] = MxData.Struct._void_to_dict(val[0])
                    else:
                        # attr_name = attr_name.replace(':', '__')
                        dico[key_name][attr_name] = val[0]
            return dico

        @staticmethod
        def _data_attributes_to_dict(attrs):
            dico = {}
            for key, val in attrs.items():
                if isinstance(val[0], np.void):
                    dico[key] = MxData.Struct._void_to_dict(val[0])
                else:
                    dico[key] = val[0]
            return dico

        @staticmethod
        def _void_to_dict(void):
            data = void.tolist()
            dtype = np.asarray(void.dtype.descr, dtype=object)  # can raise warning if not 'dtype=object'
            return dict(zip(dtype[:, 0], data))

        @staticmethod
        def convert_unit(value, unit_string):
            conv = {'Meters': (1, u.m), 'CentiMeters': (10, u.mm), 'MilliMeters': (1, u.mm), 'MicroMeters': (1, u.um),
                    'NanoMeters': (1, u.nm), 'Angstroms': (1, u.A), 'Feet': (304.8, u.mm), 'Inches': (25.4, u.mm),
                    'Mils': (25.4, u.um), 'MicroInches': (25.4, u.nm), 'NanoInches': (25.4, u.pm),
                    'Pixels': (1, u.pix), 'Waves': (1, u.wav), 'Fringes': (1, u.fr), 'FringeRadians': (1, u.frad),}
            new_unit = conv[unit_string]
            return (value * new_unit[0], new_unit[1])

        @property
        def lateral_res(self):
            unit = self.attributes['Data']['Lateral Resolution:Unit']
            value = self.attributes['Data']['Lateral Resolution:Value']
            value, unit = self.convert_unit(value, unit)
            return u.m(value, unit)

        @property
        def wavelength(self):
            unit = self.attributes['Data']['Wavelength:Unit']
            value = self.attributes['Data']['Wavelength:Value']
            value, unit = self.convert_unit(value, unit)
            return u.m(value, unit)

        @property
        def scale_factor(self):
            return self.surface_attrs['Interferometric Scale Factor']

        @property
        def obliquity_factor(self):
            return self.surface_attrs['Obliquity Factor']

        # ----overriding----
    def readfile(self, path, source=None):  # pylint: disable=unused-argument, R0914
        self.path = path
        with h5py.File(path, 'r') as datx:
            data = MxData.Struct(datx)

        self.attributes = data.attributes
        self.datetime = data.timestamp
        if isinstance(path, str):
            path = Path(path)
        path = path
        self.source = path.name
        self.header['lateral_res'] = data.lateral_res
        self.header['wavelength_in'] = data.wavelength
        # compatibilty
        self.header['ac_org_x'] = 0
        self.header['ac_org_y'] = 0
        self.header['ac_width'] = data.attributes['Data']['Camera Width:Value']
        self.header['ac_height'] = data.attributes['Data']['Camera Height:Value']
        # self.header['ac_n_buckets'] = data.attributes['Data']['ac_n_buckets']
        # self.header['ac_range'] = data.attributes['Data']['ac_range']
        # self.header['ac_n_bytes'] = data.attributes['Data']['ac_n_bytes']
        self.header['cn_width'] = data.surface_attrs['Coordinates']['Width']
        self.header['cn_height'] = data.surface_attrs['Coordinates']['Height']
        self.header['cn_org_x'] = data.surface_attrs['Coordinates']['ColumnStart']
        self.header['cn_org_y'] = data.surface_attrs['Coordinates']['RowStart']

        no_data = data.surface_attrs['No Data']
        rawdata = np.where(data.surface < no_data, data.surface, np.nan).T
        if data.surface_attrs['Unit'] == 'Fringes':
            W = data.wavelength
            S = data.scale_factor
            O = data.surface_attrs['Obliquity Factor']
            rawdata = rawdata * W * S * O
        rawdata -= np.nanmean(rawdata)
        if 'pyopticslab' in __file__:
            rawdata = np.fliplr(rawdata)  # ???
        self._raw_shape = rawdata.shape
        x = np.linspace(0, self._raw_shape[0] * self.header['lateral_res'],
                        num=self._raw_shape[0], endpoint=False, dtype=np.float64)
        y = np.linspace(0, self._raw_shape[1] * self.header['lateral_res'],
                        num=self._raw_shape[1], endpoint=False, dtype=np.float64)
        self.initial = Surface((x, y), rawdata, self.units, self.source)
        return self

        # metaX = self._axis_converter(attributes['phase']['X Converter'][0])
        # metaY = self._axis_converter(attributes['phase']['Y Converter'][0])
        # self.lateral_resolutions = (metaX['Pixels'], metaY['Pixels'])
        # # 'Meters' in metaX
        # x = np.linspace(0, self._raw_shape[0] * metaX['Pixels'], num=self._raw_shape[0], endpoint=False, dtype=np.float64)
        # y = np.linspace(0, self._raw_shape[1] * metaY['Pixels'], num=self._raw_shape[1], endpoint=False, dtype=np.float64)

    @staticmethod
    def _axis_converter(conv):
        header = []
        data = []
        for val in conv:
            if not isinstance(val, np.ndarray):
                try:
                    header.append(val.decode())
                except Exception as e:
                    print(f'_axis_converter', e)
            else:
                data = val
        dico = {}
        for key, val in zip(header, data):
            dico[key] = val
        return dico

    # def write_x_position(self, position):
    #     with h5py.File(self.path, 'r+') as hf:
    #         self._get_attributes_path(hf)
    #         for name, value in hf[self.attr_id].attrs.items():
    #             if 'coords.x_pos' in name:
    #                 hf[self.attr_id].attrs.modify(name, position)
    #

class MetroProData(ESRFOpticsLabData, Surface):
    '''MetroPro 9 data class.'''
    method = 'Fizeau Interferometer'
    instrument = "Zygo"

    def __init__(self):
        super().__init__()

        self.header_format = None
        self.header_size = None
        self.note = None

        self.datetime = None

        self._raw_shape = None
        # self.intensity = None
        # self.phase = None

        self.units = {
            'coords': u.m, 'values': u.m,
            'height': u.nm, 'angle': u.urad,
            'length': u.mm, 'radius': u.km,
            'pixel': u.mm,
        }

    def __str__(self):
        return 'Zygo surface map'

    # ----overriding----
    def readfile(self, path, source=None):  # pylint: disable=unused-argument, R0914
        self.path = path
        with open(path, 'rb') as zygo_file:

            self.datetime = dt.datetime.fromtimestamp(os.path.getmtime(path))

            if isinstance(path, str):
                path = Path(path)
            path = path
            self.source = path.name

            # Header
            magic_number = np.fromfile(zygo_file, dtype='>u4', count=1)[0]
            header_dict = header_from_txt()[magic_number]
            for key, value in header_dict.items():
                zygo_file.seek(value['pos'])
                dtype = value['dtype']
                if dtype is str:
                    size = int(value.get('size', '1'))
                    try:
                        self.header[key] = zygo_file.read(size).decode().strip('\x00')
                    except UnicodeDecodeError:
                        zygo_file.seek(value['pos'])
                        self.header[key] = zygo_file.read(size).decode('latin1').strip('\x00')
                else:
                    self.header[key] = np.fromfile(zygo_file, dtype=value['dtype'], count=1,)[0]
                    if 'f4' in value['dtype']:
                        self.header[key] = np.float64(self.header[key])

            zygo_file.seek(self.header['header_size'])
            rows = int(self.header['ac_height'])
            cols = int(self.header['ac_width'])
            size = rows * cols * self.header['ac_n_buckets']
            # self.intensity = np.resize(np.fromfile(zygo_file, dtype='>u2', count=size), (rows, cols))
            # self.intensity = np.where(self.intensity < self.header['ac_range'] + 1, self.intensity, np.nan).T
            zygo_file.seek(2*size, 1)

            rows = int(self.header['cn_width'])
            cols = int(self.header['cn_height'])
            size = rows * cols
            # self.phase = np.resize(np.fromfile(zygo_file, dtype='>i4', count=size), (cols, rows))
            # self.phase = np.where(self.phase < 2147483640, self.phase, np.nan).T
            rawdata = np.resize(np.fromfile(zygo_file, dtype='>i4', count=size), (cols, rows))
            rawdata = np.where(rawdata < 2147483640, rawdata, np.nan).T

            W = self.header['wavelength_in']
            S = self.header['intf_scale_factor']
            O = self.header['obliquity_factor']
            if self.header['phase_res'] == 0:
                R = 4096
            elif self.header['phase_res'] == 1:
                R = 32768
            elif self.header['phase_res'] == 2:
                R = 131072
            # rawdata = self.phase * W * S * O / R
            rawdata = rawdata * W * S * O / R
            rawdata -= np.nanmean(rawdata)
            if 'pyopticslab' in __file__:
                rawdata = np.fliplr(rawdata)
            self._raw_shape = rawdata.shape
            x = np.linspace(0, self._raw_shape[0] * self.header['lateral_res'],
                            num=self._raw_shape[0], endpoint=False, dtype=np.float64)
            y = np.linspace(0, self._raw_shape[1] * self.header['lateral_res'],
                            num=self._raw_shape[1], endpoint=False, dtype=np.float64)
            self.initial = Surface((x, y), rawdata, self.units, self.source)
        return self

    def writefile(self, path):
        import struct
        with open(path, 'wb') as zygo_file:
            header_dict = header_from_txt()[2283471729]  # last format version
            zygo_file.write(bytearray(4096))

            self.header['lateral_res'] = self._pixel_res[0]

            # if self.intensity is None:
            if True:
                self.header['ac_org_x'] = 0
                self.header['ac_org_y'] = 0
                # self.header['ac_width'] = 0
                # self.header['ac_height'] = 0
                self.header['ac_n_buckets'] = 0
                self.header['ac_range'] = 0
                self.header['ac_n_bytes'] = 0

            # if self.phase is None:
            if True:
                self.header['cn_org_x'] = 0
                self.header['cn_org_y'] = 0
                # self.header['cn_width'] = 0
                # self.header['cn_height'] = 0
                # self.header['cn_n_bytes'] = 0
                # self.header['phase_res'] = 0

            for key, value in header_dict.items():
                # print(key, value)
                zygo_file.seek(value['pos'])
                if key not in self.header:
                    if 'magic_number' in key:
                        val = struct.pack('>i', -2011495567)
                    elif 'header_format' in key:
                        val = struct.pack('>h', 3)
                    elif 'header_size' in key:
                        val = struct.pack('>i', 4096)
                    else:
                        continue
                else:
                    if isinstance(self.header[key], str):
                        val = bytearray(self.header[key], 'utf-8')
                    elif value['fmt'] == 'c':
                        val = bytearray(str(self.header[key]), 'utf-8')
                    else:
                        val = struct.pack(value['fmt'], self.header[key])
                zygo_file.write(val)
            zygo_file.seek(1490)
            zygo_file.write(bytearray(2606))

            # if self.intensity is not None:
            #     intensity = np.where(np.isnan(self.intensity), 65535, self.intensity).T
            #     intensity = intensity.astype('>u2').ravel().tobytes()
            #     zygo_file.seek(self.header['header_size'])
            #     zygo_file.write(intensity)
            #
            # if self.phase is not None:
            #     phase = np.where(np.isnan(self.phase), 2147483641, self.phase).T
            #     phase = phase.astype('>i4').ravel().tobytes()
            #     zygo_file.write(phase)
            self.change_values_unit(u.m)
            phase = self.values
            W = np.float64(self.header['wavelength_in'])  # in meter
            S = self.header['intf_scale_factor']
            obliquity = self.header['obliquity_factor']
            R = [4096, 32768, 131072]
            phase_res = self.header['phase_res']
            phase = phase * R[phase_res] / (W * S * obliquity)
            phase = np.where(np.isnan(phase), 2147483641, phase).T
            phase = phase.astype('>i4').ravel().tobytes()
            zygo_file.write(phase)
        return self

    # ----properties----
    # @property
    # def datetime(self):
    #     return dt.datetime.strptime(self.header['swinfo.date'], '%a %b %d %H:%M:%S %Y')
    # @property
    # def shape(self):
    #     if self.values is None:
    #         return self._raw_shape
    #     return self.values.shape
    @property
    def title(self):
        return self.header['comment']

    @property
    def mode(self):
        return 'PSI'

    @property
    def objective(self):
        return self.header['obj_name']

    @property
    def magnification(self):
        return self.header.get('gpi_enc_zoom_mag', 'no encoded zoom')
        # mag = self.header['Magnification']
        # return f'{mag:.1f} X'

    @property
    def fov(self):
        return self.header['obj_name']
        # default = '1 X'
        # return self.header.get('FOVLabel', default)

    @property
    def stage_x(self):
        return self.header['coords.x_pos']

    @property
    def stage_y(self):
        return self.header['coords.y_pos']

    @property
    def average(self):
        return self.header['phase_avg_cnt']

    @property
    def wavelength(self):
        return self.header['wavelength_in']  # m

    # @property
    # def histogram(self):
    #     mask = self.masking()
    #     return np.histogram(self.values[mask], bins=512, density=False)
    #     # plt.plot(histo[1][:-1], histo[0])


def header_from_txt():
    path_header = str(Path(Path(__file__).parent, 'metropro_headers.txt'))
    with open(path_header) as header_file:
        dico = {}
        fmt = None
        for l, line in enumerate(header_file.readlines()):
            line = line.strip()
            l = line.split()
            if 'magic_number' in line:
                fmt = int(l[4], 16)
                dico[fmt] = {}
            if fmt is None or len(l) < 5:
                continue
            try:
                size = int(l[2])
                if 'I' in l[0]:
                    dtype = 'i' + l[2]
                    if size == 1:
                        struct = 'b'
                    elif size == 2:
                        struct = 'h'
                    elif size == 4:
                        struct = 'i'
                elif 'F' in l[0]:
                    dtype = 'f' + l[2]
                    if size == 2:
                        struct = 'e'
                    elif size == 4:
                        struct = 'f'
                    elif size == 8:
                        struct = 'd'
                elif 'S' in l[0]:
                    dtype = str
                    struct = l[2] + 's'
                elif 'C' in l[0]:
                    dtype = 'B'
                    struct = 'B'
                else:
                    continue
                if 'B' in l[0]:
                    dtype = '>' + dtype
                    struct = '>' + struct
                elif 'L' in l[0]:
                    dtype = '<' + dtype
                    struct = '<' + struct
                dico[fmt][l[3]] = {
                    'dtype': dtype,
                    'fmt': struct,
                    'pos': int(l[1].split('-')[0]) - 1,
                    'size': l[2],
                    'default': l[4],
                }
            except ValueError:
                pass
        return dico
