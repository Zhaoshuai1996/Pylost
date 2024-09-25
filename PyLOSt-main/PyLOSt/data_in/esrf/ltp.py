# -*- coding: utf-8 -*-
"""Main module for the LTP data

"""

# pylint: disable=C0103, C0115, C0116

from datetime import datetime as dt
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

try:
    import swift.units as u
    import swift.utils as sw
    from esrf.data.generic import ESRFOpticsLabData, Profile, Surface
except ImportError:
    from . import units as u
    # from . import utils as sw
    from . import units as sw
    from .generic import ESRFOpticsLabData, Profile, Surface

label_x = 'mirror_coord'  # pylint: disable=invalid-name
label_y = 'mean_slopes'  # pylint: disable=invalid-name


class LTPdata(ESRFOpticsLabData, Profile):
    '''LTP data class.'''
    method = 'laser measuring system'
    instrument = "Long Trace Profiler"

    def __init__(self):
        super().__init__()
        self.slp2 = None
        self.units = {
            'coords': u.mm, 'values': u.urad,
            'height': u.nm, 'angle': u.urad,
            'length': u.mm, 'radius': u.m,
        }

        self.raw_signal = None

    # def __repr__(self):
    #     return 'repr'

    def __str__(self):
        return 'LTP 1D scan'

    # ----overriding----
    def readfile(self, path, source=None):
        data = None
        if isinstance(path, str) and '\n' in path:
            data = path  # data is passed as argument
            if source is None:
                source = 'buffer'
        else:
            try:
                with open(path) as slpfile:
                    path = Path(path)
                    self.path = path
                    source = path.name
                    data = slpfile.read()
            except:
                print('Error reading LTP data (No such file).')
                return None
            # pass
        self.source = source
        if data is None:
            return None
        if '[' in data[:10]:
            # try:
            self._parseslp2(data)
        else:
            # except ValueError:
            self._parse_slp(data)
        return self

    def writefile(self, path, ext='slp'):
        if isinstance(path, str):
            path = Path(path)
        # return self._build_slp()
        if 'slp2' in ext.lower():
            buf = self._build_slp2()
            with open(path, 'w') as ltp_file:
                ltp_file.write(buf)
        path = Path(path.parent, path.stem + '.slp')
        buf = self._build_slp()
        with open(path, 'w') as ltp_file:
            ltp_file.write(buf)
        return buf

    # ----properties----
    @property
    def datetime(self):
        date_time = self.header['legacy_header']['date_time']
        return dt.strptime(date_time, '%d/%m/%Y @ %H:%M:%S')

    @property
    def archive_path(self):
        return str(self.path.parent) + '\\' + self.path.stem + '_fwd-bwd.7z'

    @property
    def raw_path(self):
        return str(self.path.parent) + '\\' + self.path.stem + '.raw'

    # ----write slp/slp2 file----
    def _build_slp(self):
        header = self.header['legacy_header']
        buf = header.get('comment', '') + '\n'
        buf = buf + header.get('date_time', '') + '\n'
        buf = buf + header.get('reference', 'No ref/') + '/'
        buf = buf + header.get('gravity', '') + '\n'
        focal = header.get('focal_length', '0')
        buf = buf + f'{focal:.6f}' + '\n'
        buf = buf + f'{self.mean_step:.10f}' + '\n'
        buf = buf + f'{len(self.values)}' + '\n'
        for i, val in enumerate(self.values):
            buf = buf + f'{self.coords[i]:.10f}\t{val:.10f}' + '\n'
        return buf

    def _build_slp2(self):
        def header_to_buf(buf, dico):
            for key, value in dico.items():
                if isinstance(value, dict):
                    if 'loaded_data' in key:
                        header = list(value.keys())
                        header = ' '.join(['%s'] * len(header)) % tuple(header)
                        buf = buf + '[data]\ndata_header = ' + f'"{header}"\n'
                        buf = buf + 'Slopes_Xmin = "\n'
                        data = np.asfarray(list(value.values()))
                        for pos in data.T:
                            buf = buf + '\t'.join(['%.4f'] * len(pos)) % tuple(pos) + '\n'
                    else:
                        buf = buf + f'[{key}]\n'
                        buf = header_to_buf(buf, value)
                        buf = buf + '\n'
                else:
                    if isinstance(value, float):
                        if np.isnan(value):
                            value = 'NaN'
                        elif 'SLP2 version' in key:
                            value = f'{value:.1f}'
                        else:
                            value = f'{value:.3f}'
                    elif isinstance(value, np.ndarray):
                        value = ' '.join(['%.4f'] * len(value)) % tuple(value)
                    buf = buf + f'{key} = "{value}"\n'
            return buf

        buf = header_to_buf('', self.header)
        buf = buf + '"'
        return buf

    # ----read slp/slp2 file----
    def _parse_slp(self, buf):
        """Build the slp nested dictionary with section, key and values.
        """
        self.slp2 = False
        self.header['legacy_header'] = {}
        data_array = []
        header = ['comment', 'date_time', 'reference/gravity',
                  'focal_length', 'step', 'nb_acquisitions']
        for num, line in enumerate(buf.splitlines()):
            value = self._val_interp(line)
            if not line:
                continue
            if num < len(header):
                if num < 3:
                    value = line.strip()
                if num == 2:
                    twice = line.split('/')
                    self.header['legacy_header']['reference'] = twice[0]
                    try:
                        self.header['legacy_header']['gravity'] = twice[1]
                    except IndexError:
                        self._raisevaluerror(3)
                else:
                    self.header['legacy_header'][header[num]] = value
                continue
            if len(value) > 1:
                try:
                    data_array.append(value)
                except ValueError:
                    self._raisevaluerror(3)
        data_header = [label_x, label_y]
        self._finalize(data_array, data_header, errorcodes=(3, 4, 5))

    def _parseslp2(self, buf):
        """Build the slp2 nested dictionary with section, key and values.
        """
        if buf.find('legacy_header') < 0 & buf.find('general_header') < 0:
            self._raisevaluerror(6)
        self.slp2 = True
        section = None
        data_array = []
        data_header = ['mirror_coord', 'mean_slopes']  # 1st version
        for line in buf.splitlines():
            # almost empty line ?
            if len(line) < 3:
                continue
            # section ?
            if line.endswith(']'):
                section = line.strip('[]')
                if section != 'data':
                    if section == 'general_header':  # 1st version
                        section = 'legacy_header'
                    self.header[section] = {}
                continue
            if section:
                # key = value ?
                if '=' in line:
                    (key, value) = line.split('=')
                    key = key.strip()
                    value = value.split('"')[1]
                    # valid ?
                    if len(value) > 0 or 'comment' in key.lower():
                        if section == 'data' or section == 'Slopes':  # pylint: disable=R1714
                            data_header = self._val_interp(value).split()
                        else:
                            self.header[section][key] = self._val_interp(value)
                    continue
                if section == 'data':
                    data_array.append(self._val_interp(line))
        self._finalize(data_array, data_header, errorcodes=7)

    def _finalize(self, data_array, data_header, errorcodes=2):
        if isinstance(errorcodes, int):
            errorcodes = (errorcodes, errorcodes, errorcodes)
        format_error, value_error, data_error = errorcodes
        try:
            data_header[0] = label_x
            data_header[1] = label_y
            num_acq = self.header['legacy_header']['nb_acquisitions']
        except KeyError:
            self._raisevaluerror(format_error)
        if num_acq == 0:
            self._raisevaluerror(value_error)
        if num_acq == len(data_array) + 1:
            # bug in firsts static slp2 files
            if (data_array[-1] + data_array[-2])[0] < 1e-3:
                data_array = data_array[:-1]
            self._raisevaluerror(data_error)
        self.header['legacy_header']['nb_acquisitions'] = len(data_array)
        data_array = np.asfarray(data_array).T
        num_entries = data_array.shape[0]
        self.header['loaded_data'] = {}
        for i, key in enumerate(data_header):
            if i > num_entries - 1:
                break
            self.header['loaded_data'][key] = data_array[i]
        self.header_dict_to_class()
        self.initial = Profile(data_array[0], data_array[1],
                               self.units, self.source)

    @staticmethod
    def _val_interp(string):
        '''Return variable type from input string'''
        string = string.strip()
        if len(string) < 1:  # empty string
            return ''
        array = string.split()
        try:
            dtype = 'float64'
            if np.all(np.char.isdigit(string.rsplit('-')[-1])):
                dtype = 'int'
            array = np.asarray(array, dtype=dtype)
        except ValueError:
            return string
        if len(array) > 1:
            return array
        return array[0]

    @staticmethod
    def _raisevaluerror(code, string=''):
        msg = ['Not the right extension!',
               f'source {string} has not the right extension!',
               'Data not valid!',
               'Wrong slp format',
               'Wrong slp format or error on slp data!',
               'Error on slp data!',
               'Wrong slp2 format',
               'Error on slp2 data!'
               ]
        raise ValueError(msg[code])

    # ----archive data----
    @staticmethod
    def read_archive_data(archive_path, include_raw=True, view=False, verbose=True):
        data = {'FWD': {}, 'BWD': {}}
        # if isinstance(archive_path, str):
        #     archive_path = Path(archive_path)
        file_type = ('slp', 'slp2')
        if include_raw:
            file_type = file_type + ('raw',)
        with open(archive_path, 'rb') as archive:
            for dataname, buffer in sw.list_entries(archive,
                                                    key_word='WD_',
                                                    extension=file_type):
                split = dataname.rpartition('.')
                extension = split[-1]
                direction = split[0].rpartition('_scan-')[-1]
                direction, _, number = direction.rpartition('_')
                if number not in data[direction]:
                    data[direction][number] = {'slp': None, 'slp2': None, 'raw': None}
                value = None
                if 'raw' in extension.lower():
                    try:
                        prescan = data[direction][number]['slp2'].scan_header.prescan
                        step = np.abs(data[direction][number]['slp2'].legacy_header.step)
                        prescan_int = prescan / step
                    except AttributeError:
                        prescan_int = 0
                    value = LTPdata.RawData(buffer, prescan_int)
                else:
                    value = LTPdata.read(buffer, dataname, verbose=True)
                    if 'slp2' in extension.lower():
                        prescan_int = value.scan_header.prescan / np.abs(value.legacy_header.step)
                        try:
                            data[direction][number]['raw'].prescan_int = prescan_int
                        except AttributeError:
                            pass
                data[direction][number][extension] = value
            if view:
                LTPdata.data_view(data)
            return data

    @staticmethod
    def data_view(data):
        import matplotlib.pyplot as plt
        plt.ion()
        # plt.show()
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        for a, direction in enumerate(['FWD', 'BWD']):
            slopes = []
            for key, values in data[direction].items():
                values['slp2'].polynomial_removal(2)
                slopes.append(values['slp2'].values)
            slopes = np.asfarray(slopes)
            axs[a].plot(slopes.T, linewidth=1)
        # plt.draw()
        plt.pause(0.001)
        # slopes = np.asfarray(slopes).T
        # y = np.linspace(1, slopes.shape[1], num=slopes.shape[1])[::-1]
        # sub = Surface((values['slp2'].coords, y), slopes, values['slp2'].units)
        # sub.plot(title=direction, reverse_y=True, base='rq', factor=12, cmap='pink')

    # ----signal analysis----
    @staticmethod
    def exclude_scans(data, order=2, threshold=0.150, use_xmin=None, focal_length=812.31):
        '''Reconstruct profile of a LTP scan from its archive by applying
        an exclusive threshold on polynomial fit of the FWD/BWD scans.

            Return the updated profile after out of range fwd/bwd scans exclusion and the index of scans excluded

            Parameters :
                'data' : dictionary containing the full set of FWD/BWD scans.

                'order' : int   (default 2)  polynomial order

                'threshold' : float   (default 0.150)  exclusion threshold

                'use_xmin' : boolean   (default None)  convert xmin to slopes according to the 'focal_length'

                'focal_length' : float   (default 812.31)
        '''
        slopes_average = []
        xmins_average = []
        excluded = {'FWD': list(), 'BWD': list()}
        deviations = {'FWD': np.nan, 'BWD': np.nan}
        for direction in ('FWD', 'BWD'):
            slopes_init = []
            slopes_temp = []
            xmins = []
            for key, values in data[direction].items():
                xmin = (values['slp2'].loaded_data.SUT_Xmin, values['slp2'].loaded_data.REF_Xmin)
                if use_xmin is not None:
                    pixel_size = 7
                    if values['slp2'].loaded_data.REF_Xmin[0] < 1024:
                        pixel_size = 25  # old detector
                    coef = 1e3 * pixel_size / (2 * focal_length)
                    if use_xmin:
                        values['slp2'].values = sum(xmin) * coef
                    else:
                        values['slp2'].values = xmin[0] * coef
                        values['slp2'].legacy_header.reference = 'No ref'
                slopes_init.append(values['slp2'].values.copy())
                xmins.append(xmin)
                values['slp2'].polynomial_removal(order)
                slopes_temp.append(values['slp2'].values)
                continue
            slopes_init = np.asfarray(slopes_init)
            slopes_temp = np.asfarray(slopes_temp)
            deviation = [threshold, ]
            while max(deviation) >= threshold:
                ave = slopes_temp - slopes_temp.mean(axis=0)
                deviation = ave.std(axis=1)
                deletion = deviation.argmax()
                if deviation.max() > threshold:
                    slopes_temp = np.delete(slopes_temp, deletion, axis=0)
                    slopes_init = np.delete(slopes_init, deletion, axis=0)
                    xmins.pop(deletion)
                    print(f'            scan {direction} {deletion + 1:02i} removed')
                    excluded[direction].append(deletion + 1)
            slopes_average.append(slopes_init.mean(axis=0))
            xmins_average.append(np.asfarray(xmins).mean(axis=0))
            deviations[direction] = deviation.max()
        # slopes = np.asfarray(slopes_average).mean(axis=0)
        # xmins = np.asfarray(xmins_average).mean(axis=0)
        return np.asfarray(slopes_average), np.asfarray(xmins_average), excluded, deviations

    # ----fwd/bwd----
    @staticmethod
    def process_fwd_bwd(root_file, exclude_ref=False, verbose=False, show=False, **kwargs):
        '''Reconstruct profile of a LTP scan from its FWD/BWD archive.

            Return the new averaged profile after out of range fwd/bwd scans exclusion.

            Parameters :
                'root_file' : path to the LTP scan.

                'exclude_ref' : boolean   (default False)

                'verbose' : boolean   (default False)

                'show' : boolean   (default False)  display plots

                'kwargs':

                    'order' : int   (default 2)  polynomial order

                    'threshold' : float   (default 0.150)  exclusion threshold

                    'use_xmin' : boolean   (default None)  convert xmin to slopes
        '''
        scan = LTPdata.read(root_file, verbose=True)
        if scan is None:
            return None
        include_raw = exclude_ref or kwargs.get('use_xmin') is not None
        try:
            print('    reading attached archive')
            data = scan.read_archive_data(scan.archive_path, include_raw=include_raw, verbose=verbose)
        except:
            print('      ERROR: attached archive not found --> skip')
            return None
        if show:
            LTPdata.data_view(data)
        data['ID'] = scan.source
        focal_length = scan.legacy_header.focal_length
        values, xmins, idx, dev = scan.exclude_scans(data, focal_length=focal_length, **kwargs)
        scan.values = values.mean(axis=0)
        scan.header['loaded_data']['mean_slopes'] = scan.values.copy()
        xmins = xmins.mean(axis=0)
        scan.header['loaded_data']['mean_SUT_Xmin'] = xmins[0]
        scan.header['loaded_data']['mean_REF_Xmin'] = xmins[1]
        if exclude_ref:
            scan.legacy_header.reference = 'No ref '
        else:
            scan.legacy_header.reference = 'Reference subtracted '
        if show:
            cpy = data.copy()
            for direction, removed in idx.items():
                for i in removed:
                    cpy[direction].pop(f'{i:02d}')
            LTPdata.data_view(cpy)
            ans = input('(A)ccept, (R)eload, (I)gnore ? >> ')
            if 'r' in ans.lower():
                return False
            elif 'i' in ans.lower():
                return None
        if scan.slp2:
            scan.header['fwd-bwd_header']['detrending_order'] = kwargs['order']
            scan.header['fwd-bwd_header']['exclusion_treshold'] = kwargs['threshold']
            scan.header['fwd-bwd_header']['meanRMS_BWD'] = dev['BWD']
            scan.header['fwd-bwd_header']['nb_valid_BWD'] = scan.scan_header.average - len(idx['BWD'])
            scan.header['fwd-bwd_header']['meanRMS_FWD'] = dev['FWD']
            scan.header['fwd-bwd_header']['nb_valid_FWD'] = scan.scan_header.average - len(idx['FWD'])
            scan.header_dict_to_class()
        return scan

    @staticmethod
    def process_folder(startfolder, overwrite=False, **kwargs):
        path = sw.choosefile(startpath=startfolder, filters=('LTP measurement', 'slp'))
        if path is None:
            return None
        scans = {}
        slp2 = False
        for file in sw.searchfiles(path,
                                   key_word=path.stem.rpartition('_')[0],
                                   extension='slp2',
                                   recursive=False,
                                   empty=False):
            while slp2 is False:
                slp2 = LTPdata.process_fwd_bwd(file, verbose=False, show=False, **kwargs)
            if slp2 is None:
                slp2 = False
                continue
            scans[path.stem] = slp2
            if overwrite is not None:
                if not overwrite:
                    path = Path(str(file.parent) + '\\reprocessed\\' + file.name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                buf = LTPdata.writefile(path, file.suffix)
            slp2 = False
            print('')
        return scans

    # ----stitching----
    @staticmethod
    def process_stitching_scans(scans=None, fwd_bwd=False, **kwargs):
        '''Stitch subscans from scans dictionary or from FWD/BWD archive.

            Return the stitched values numpy array, the stitch LTPdata object and the scans dictionary.

            Parameters :
                'scan' : dictionary containing the LTPdata subscans.
                         'None' if 'fwd_bwd' set to True.
                         (default None)

                'fwd_bwd' : boolean   (default False)
                            if True, use 'root_file' as path to archive

                'kwargs' : additional options in the fwd/bwd processing
                    'root_file' : str or pathlib.Path for archive path

                    'order' : int   (default 2)  exclusion polynomial order

                    'threshold' : float   (default 0.150)  exclusion threshold

                    'use_xmin' : boolean   (default None)

                    'exclude_ref' : boolean   (default False)
        '''
        if fwd_bwd:
            scans = {}
            root_file = kwargs.pop('root_file', '')
            if root_file is None:
                return None
            if isinstance(root_file, str):
                root_file = Path(root_file)
            for path in sw.searchfiles(str(root_file.parent) + '\\StFiles',
                                       key_word=root_file.stem,
                                       extension='slp',
                                       recursive=False,
                                       empty=False):
                sub = LTPdata.process_fwd_bwd(path, **kwargs)
                if sub is None:
                    return None
                st = sub.path.stem.split('_')
                # run = st[-2].split('run')[-1]
                st = st[-1].split('st')[-1]
                scans[st] = sub
        if len(scans) == 0:
            return None
        import pandas as pd
        scan_df = []
        result = pd.DataFrame({'coords': [], 'slopes': [], })
        result.set_index('coords', inplace=True)
        for key, scan in scans.items():
            current = pd.DataFrame({'coords': np.around(scan.coords, 3), key: scan.values, })
            current.set_index('coords', inplace=True)
            scan_df.append(current)

        result = pd.concat(scan_df, axis=1, sort=True)
        result.plot(legend=False)

        for s in range(1, result.shape[1]):
            diff = (result.iloc[:, s - 1] - result.iloc[:, s]).mean()
            result.iloc[:, s] = result.iloc[:, s] + diff

        # result.sort_index(ascending=False, inplace=True)
        result.plot(legend=False)

        stitched_scan = result.mean(axis=1)
        stitched_scan.sort_values(ascending=False, inplace=True)
        stitched_scan.index = np.around(np.nanmean(stitched_scan.index) - stitched_scan.index, 3)
        # stitched_scan.index = np.around(stitched_scan.index - np.nanmean(stitched_scan.index), 3)

        buffer = 'stitched data (python)\n'
        buffer = buffer + dt.strftime(scan.datetime, '%d/%m/%Y @ %H:%M:%S\n')
        buffer = buffer + f'{scan.legacy_header.reference}/{scan.legacy_header.gravity}\n'
        buffer = buffer + f'{scan.legacy_header.focal_length:.6f}\n'
        step = np.diff(stitched_scan.index).mean()
        buffer = buffer + f'{step}\n'
        nb_pts = stitched_scan.shape[0]
        buffer = buffer + f'{nb_pts}\n'
        for i, val in enumerate(stitched_scan.values):
            buffer = buffer + f'{stitched_scan.index[i]:.10f}\t{val:.10f}' + '\n'

        return stitched_scan, LTPdata.read(buffer), scans

    class RawData():
        '''LTP raw data class.'''

        def __init__(self, buffer, prescan_int=None):
            self.integration_time = None
            self.scan_speed = None
            self.threshold = None
            self.translation = None
            self.SUT = None
            self.SUT_pos = None
            self.REF = None
            self.REF_pos = None
            self.qsys_ltp = None
            self._prescan_int = int(prescan_int)
            self.parse_raw_data(buffer)

        @property
        def prescan_int(self):
            return self._prescan_int

        @prescan_int.setter
        def prescan_int(self, prescan_int):
            self._prescan_int = int(prescan_int)

        @property
        def measurement_position(self):
            prescan = self.prescan_int
            if prescan > 0 and self.qsys_ltp:
                prescan = prescan - 1
            return self.translation[prescan:-prescan]

        @property
        def measurement_raw(self):
            prescan = self.prescan_int
            if prescan > 0 and self.qsys_ltp:
                prescan = prescan - 1
            return (self.SUT[prescan:-prescan], self.REF[prescan:-prescan])

        @property
        def measurement_xmin(self):
            xmins = []
            for patterns in zip(*self.measurement_raw):
                xmins.append(self.intercorrelation(patterns))
            xmins = np.asfarray(xmins)
            prescan = self.prescan_int
            if prescan > 0 and self.qsys_ltp:
                prescan = prescan - 1
            pos = (self.SUT_pos[prescan:-prescan], self.REF_pos[prescan:-prescan])
            return xmins + np.asfarray(pos).T

        @property
        def measurement_peaks(self):
            peaks = []
            for patterns in zip(*self.measurement_raw):
                peaks.append(self.peaks_detection(patterns))
            return peaks

        def measurement_slopes(self, focal_length=810.0, include_ref=True):
            pixel_size = 7
            if self.measurement_xmin[0][1] < 1024:
                pixel_size = 25  # old detector
            coef = 1e3 * pixel_size / (2 * focal_length)
            if include_ref:
                xmin = self.measurement_xmin.sum(axis=1)
            else:
                xmin = self.measurement_xmin[:, 0]
            return xmin * coef

        def parse_raw_data(self, string):
            raw = None
            if isinstance(string, str):
                if '\n' in string:
                    raw = string  # data is passed as argument
                    print('Reading LTP raw data.')
                else:
                    try:
                        with open(string) as file:
                            raw = file.read()
                            print(f'Reading LTP raw data: {string}.')
                    except:
                        print('Error reading raw data (No such file).')
                        return None
            # raw_signal = {}
            data_array = []
            for num, line in enumerate(raw.splitlines()):
                array = np.asfarray(line.strip().split())
                if num == 0:
                    self.threshold = array[0]
                    self.integration_time = array[1]
                    self.scan_speed = array[2]
                    continue
                data_array.append(array)
            data_array = np.asfarray(data_array)
            self.translation = data_array[:, 0]
            self.SUT_pos = data_array[:, 1]
            self.REF_pos = data_array[:, 2]
            window = int((data_array.shape[1] - 3) / 2)
            self.SUT = data_array[:, 3:3 + window]
            self.REF = data_array[:, 3 + window:]
            self.qsys_ltp = len(array) > 512

        @staticmethod
        def peaks_detection(patterns, threshold=20000, peak_separation=48):
            all_peaks = []
            for pattern in patterns:
                peaks = {}
                p = np.where(pattern > threshold, pattern - threshold, 0)
                pos, _ = find_peaks(p, distance=peak_separation)
                if len(pos) < 2:
                    return None
                window = pos[1] - pos[0]
                half = int(window / 2)
                xmin_int = pattern[pos[0]:pos[1]].argmin() + pos[0]
                peaks['Xmin_int'] = xmin_int
                p1 = pattern[pos[0] - half:pos[0] + half]
                mask = np.where(p1 > 0, True, False)
                domain = np.linspace(pos[0] - half, pos[0] + half, num=half * 2, endpoint=False)
                P1 = np.polynomial.Polynomial.fit(domain[mask], p1[mask], deg=2)
                peaks['p1_pos'] = P1.deriv().roots()[0] + pos[0]
                peaks['p1_hgt'] = P1.linspace()[1].max()
                p2 = pattern[pos[1] - half:pos[1] + half]
                mask = np.where(p1 > 0, True, False)
                domain = np.linspace(pos[1] - half, pos[1] + half, num=half * 2, endpoint=False)
                P2 = np.polynomial.Polynomial.fit(domain[mask], p2[mask], deg=2)
                peaks['p2_pos'] = P2.deriv().roots()[0] + pos[1]
                peaks['p2_hgt'] = P2.linspace()[1].max()
                peaks['pk_sep'] = peaks['p2_pos'] - peaks['p1_pos']
                all_peaks.append(peaks)
            if len(all_peaks) == 1:
                return peaks, None
            return all_peaks

        @staticmethod
        def intercorrelation(patterns):
            xmins = []
            for pattern in patterns:
                # if pattern.max()
                # m = ceil(log(2*pattern.argmax())/log(2))
                # n = 2**m
                # n = (n<len(pattern)+2) ? 2**(m+1),n
                arr = np.concatenate((pattern, np.zeros((256,))))
                a = np.fft.fft(arr)
                b = np.concatenate((a[:256], np.zeros((1536,), dtype=np.csingle), a[256:]))
                c = abs(np.fft.ifft(b * b))
                threshold = c.max() * 0.9
                p = np.where(c > threshold, c - threshold, 0)
                pos, _ = find_peaks(p, distance=48)
                mask = np.where(p > 0, True, False)
                xmin = np.nan
                if len(p[mask]) != 0:
                    pixel = np.linspace(0, 2048, num=2048, endpoint=False)[mask]
                    coef = np.polynomial.polynomial.polyfit(pixel, p[mask], deg=2)
                    xmin = (-coef[1] / (2 * coef[2])) / 8  # + pixel[0]
                xmins.append(xmin)
            if len(xmins) == 1:
                return xmin, None
            return xmins

        @staticmethod
        def folder_in_memory(folder, include_raw=True):
            print('Select a folder.')
            folder = sw.choosefile(startpath=folder, filters=('7Zip archive', '7z'))
            if folder is None:
                print('Canceled.')
                return
            if folder.is_file():
                folder = folder.parent
            print(f'Working directory: {folder}')
            database = {}
            for path in sw.searchfiles(folder,
                                       key_word='_fwd-bwd',
                                       extension='7z',
                                       recursive=False,
                                       empty=False):
                scans = LTPdata.read_archive_data(path, include_raw=include_raw)
                database[path.name] = scans
            print('\nfolder_in_memory terminated successfully.')
            return database

        @staticmethod
        def data2csv(folder=None, output_file=None, add_temp=True, add_xmin=True, add_peaks=True, recursive=False):
            if folder is None:
                print('Select a folder.')
                folder = sw.choosefile(startpath=folder, filters=('7Zip archive', '7z'))
                if folder is None:
                    print('Canceled.')
                    return
            if not isinstance(folder, Path):
                folder = Path(folder)
            if folder.is_file():
                folder = folder.parent
            print(f'Working directory: {folder}')

            # title = 'scan , date time, excel time'
            title = 'scan, start time'
            if add_temp:
                title = title + ', start temp 1, start temp 2, start temp 3, start temp 4, start temp 5, start temp 6, start temp 7, start temp 8'
            if add_xmin:
                title = title + ', SUT Xmin begin, SUT Xmin midscan, SUT Xmin end, REF Xmin begin, REF Xmin midscan, REF Xmin end'
            if add_peaks:
                title = title + ', SUT peak 1 height midscan, SUT peak 2 height midscan, SUT peak separation midscan, REF peak 1 height midscan, REF peak 2 height midscan, REF peak separation midscan'
            title = title + ', stop time, stop temp 1, stop temp 2, stop temp 3, stop temp 4, stop temp 5, stop temp 6, stop temp 7, stop temp 8'

            folder_path = None
            for path in sw.searchfiles(folder,
                                       key_word='_fwd-bwd',
                                       extension='7z',
                                       recursive=recursive,
                                       empty=False):
                if folder_path is not path.parent:
                    folder_path = path.parent
                    output_file = str(folder_path.parent) + f'\\{folder_path.name}.csv'
                exist = Path(output_file).exists()
                with open(output_file, 'a+') as output:
                    if not exist:
                        output.write(title + '\n')
                    scans = LTPdata.read_archive_data(path, include_raw=add_peaks)
                    size = max(len(scans['FWD']), len(scans['BWD']))
                    for scan in range(1, size + 1):
                        num = f'{scan:02d}'
                        for direction in ('FWD', 'BWD'):
                            string = ''
                            try:
                                slp2 = scans[direction][num]['slp2']
                                string = string + f'{slp2.source}'
                                # datetime = slp2.snapshots_header.time_start.split('@')
                                # string = string + f', {datetime[0].strip()} {datetime[1].strip()}'
                                datetime = dt.strptime(slp2.snapshots_header.time_start, '%d/%m/%Y @ %H:%M:%S')
                                string = string + f', {sw.dt2excel(datetime):.5f}'
                            except:
                                continue
                            if add_temp:
                                try:
                                    temp = slp2.snapshots_header.temperature_start
                                except:
                                    temp = [np.nan for t in range(8)]
                                temp = ', '.join(['%.4f'] * 8) % tuple(temp)
                                string = string + ', ' + temp
                            midscan = int(slp2.legacy_header.nb_acquisitions / 2)
                            if add_xmin:
                                sut = slp2.loaded_data.SUT_Xmin[0]
                                string = string + f', {sut:.2f}'
                                sut = slp2.loaded_data.SUT_Xmin[midscan]
                                string = string + f', {sut:.2f}'
                                sut = slp2.loaded_data.SUT_Xmin[-1]
                                string = string + f', {sut:.2f}'
                                ref = slp2.loaded_data.REF_Xmin[0]
                                string = string + f', {ref:.2f}'
                                ref = slp2.loaded_data.REF_Xmin[midscan]
                                string = string + f', {ref:.2f}'
                                ref = slp2.loaded_data.REF_Xmin[-1]
                                string = string + f', {ref:.2f}'
                            if add_peaks:
                                try:
                                    sut, ref = scans[direction][num]['raw'].measurement_peaks[midscan]
                                    amp = (sut['p1_hgt'], sut['p2_hgt'], sut['pk_sep'],
                                           ref['p1_hgt'], ref['p2_hgt'], ref['pk_sep'])
                                except:
                                    amp = [np.nan for t in range(6)]
                                try:
                                    amp = ', '.join(['%.2f'] * 6) % tuple(amp)
                                except:
                                    pass
                                string = string + ', ' + amp

                            datetime = dt.strptime(slp2.snapshots_header.time_end, '%d/%m/%Y @ %H:%M:%S')
                            string = string + f', {sw.dt2excel(datetime):.5f}'
                            if add_temp:
                                try:
                                    temp = slp2.snapshots_header.temperature_end
                                except:
                                    temp = [np.nan for t in range(8)]
                                temp = ', '.join(['%.4f'] * 8) % tuple(temp)
                                string = string + ', ' + temp

                            output.write(string + '\n')
                    continue
                # return
