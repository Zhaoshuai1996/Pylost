# coding=utf-8
import h5py

import numpy as np
from PyQt5.QtWidgets import QLineEdit, QHBoxLayout, QWidget, QLabel, QSizePolicy as Policy

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.ow_filereaders import H5Reader, H5TreeDialog
from pylost_widgets.util.util_functions import format_nanarr


class MINTNexusTreeDialog(H5TreeDialog):
    """Dialog for loading HDF5 files"""

    def __init__(self, parent=None, filename=''):
        super(MINTNexusTreeDialog, self).__init__(parent, filename)
        widget = QWidget()
        hbox = QHBoxLayout()
        widget.setLayout(hbox)
        widget.setSizePolicy(Policy.Fixed, Policy.Fixed)

        label = QLabel('Scale height : ')
        label.setSizePolicy(Policy.Fixed, Policy.Fixed)
        hbox.addWidget(label)
        self.scale_heights = QLineEdit('255.5')
        self.scale_heights.setSizePolicy(Policy.Fixed, Policy.Fixed)
        self.scale_heights.setFixedWidth(100)
        hbox.addWidget(self.scale_heights)

        label = QLabel('Scale motor X : ')
        label.setSizePolicy(Policy.Fixed, Policy.Fixed)
        hbox.addWidget(label)
        self.scale_mx = QLineEdit('-1')
        self.scale_mx.setSizePolicy(Policy.Fixed, Policy.Fixed)
        self.scale_mx.setFixedWidth(50)
        hbox.addWidget(self.scale_mx)

        label = QLabel('Scale motor Y : ')
        label.setSizePolicy(Policy.Fixed, Policy.Fixed)
        hbox.addWidget(label)
        self.scale_my = QLineEdit('1')
        self.scale_my.setFixedWidth(50)
        hbox.addWidget(self.scale_my)
        self.lt.addWidget(widget)

    def get_params(self):
        try:
            h = self.scale_heights.text()
            x = self.scale_mx.text()
            y = self.scale_my.text()
            if h != '':
                h = float(h)
            if x != '':
                x = float(x)
            if y != '':
                y = float(y)
            return h, x, y
        except Exception:
            raise Exception('Error reading scaling factors')


class MINTNexusReader(H5Reader):
    """File reader for hdf5 formats"""
    EXTENSIONS = ('.nxs',)
    DESCRIPTION = 'Nexus file reader for the SOLEIL MINT instrument data'
    SUPPORT_COMPRESSED = False
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 2
    exclude_interferograms = True
    clear_output_before_loading = True
    scale_height = 1.0
    scale_motor_x = 1.0
    scale_motor_y = 1.0

    def read(self, dialog=None):
        """
        Reader for nexus file formats. It opens a dialog and provides option to load partially a single group in file.

        :return: Formatted file data as dictionary
        :rtype: dict
        """
        dialog = MINTNexusTreeDialog(None, self.filename)
        dialog.setWindowTitle('Load MINT nexus file')
        data = super(MINTNexusReader, self).read(dialog=dialog)
        self.scale_height, self.scale_motor_x, self.scale_motor_y = data['scale_height'], data['scale_motor_x'], data[
            'scale_motor_y'] = dialog.get_params()
        return data

    def dump_to_dict(self,
                     h5f,
                     path="/",
                     exclude_names=None,
                     asarray=True,
                     dereference_links=True,
                     include_attributes=False,
                     errors='raise'):
        if self.exclude_interferograms:
            exclude_names = self.exclude_names = []
            h5f.visititems(self.get_exclude_names)
        return super(MINTNexusReader, self).dump_to_dict(h5f, path, exclude_names, asarray=False,
                                                         dereference_links=dereference_links,
                                                         include_attributes=include_attributes,
                                                         errors=errors)

    def get_exclude_names(self, name, node):
        if node.attrs.get('NX_class') == b'NXinterferogram_collection':
            self.exclude_names.append(name.split('/')[-1])
        return None

    @classmethod
    def write_file(cls, filename, data):
        """
        H5 writer to save dictionary data.

        :param filename: Save file name
        :type filename: str
        :param data: Dictionary data
        :type data: dict
        """
        H5Reader.write_file(filename, data)

    def load_first_entry(self, data, h5f):
        for key in h5f.keys():
            if h5f[key].attrs.get('NX_class') == b'NXentry':
                ret_data = {'module': 'scan_data', 'entry': key}
                ret_data['scan_data'] = self.get_scans(data[key], h5f[key])
                return ret_data

    def get_scans(self, data, h5entry):
        ret_data = {}
        for key in h5entry.keys():
            if h5entry[key].attrs.get('NX_class') == b'NXscan':
                ret_data[key] = self.get_scan(data[key], h5entry[key])
        return ret_data

    def get_scan(self, data, h5scan):
        ret_data = {}
        height = None
        cam_size = None
        pix_size = None
        motor_X = None
        motor_Y = None
        motor_Z = None
        for skey in h5scan.keys():
            if h5scan[skey].attrs.get('NX_class') == b'NXscanData':
                h5scandata = h5scan[skey]
                for sdkey in h5scandata:
                    if h5scandata[sdkey].attrs.get('NX_class') == b'NXwavefront_collection':
                        h5wfs = h5scandata[sdkey]
                        # Assuming all wavefront are of the same size, as there is no start position info inside NXwavefront
                        # First calculate the shape required for 3D subaperture array, and then build array from already loaded 'data'.
                        # If image size and count is available in NXparameters, maybe we can use them?
                        shp = [0, 0, 0]
                        for wkey in h5wfs:
                            if h5wfs[wkey].attrs.get('NX_class') == b'NXwavefront':
                                shp[0] += 1
                                if shp[1] == 0 or shp[2] == 0:
                                    shp[1:] = h5wfs[wkey]['unwrpdPhase'].shape
                        height = np.full(shp, dtype=np.float32, fill_value=np.nan)
                        i = 0
                        for wkey in h5wfs:
                            if h5wfs[wkey].attrs.get('NX_class') == b'NXwavefront':
                                height[i] = data[skey][sdkey][wkey]['unwrpdPhase']
                                i += 1
                        break
            elif h5scan[skey].attrs.get('NX_class') == b'NXinstrument':
                h5instr = h5scan[skey]
                data_instr = data[skey]
                for ikey in h5instr.keys():
                    if h5instr[ikey].attrs.get('NX_class') == b'NXdetector':
                        cam_size = [data_instr[ikey]['y_dim'][0], data_instr[ikey]['x_dim'][0]]
                        pix_size = [data_instr[ikey]['y_pixel_size'][0], data_instr[ikey]['x_pixel_size'][0]]
                    if h5instr[ikey].attrs.get('NX_class') == b'NXactuator':
                        motor_X = data_instr[ikey].get('sample_X', None)
                        motor_Y = data_instr[ikey].get('sample_Y', None)
                        motor_Z = data_instr[ikey].get('sample_Z', None)

        if height is not None:
            motors = []
            height = self.scale_height * height
            if motor_X is not None:
                motor_X = self.scale_motor_x * motor_X
                motors += [{'name': 'motor_X', 'values': format_nanarr(motor_X), 'axis': [-3], 'unit': 'mm'}]
            if motor_Y is not None:
                motor_Y = self.scale_motor_y * motor_Y
                motors += [{'name': 'motor_Y', 'values': format_nanarr(motor_Y), 'axis': [-3], 'unit': 'mm'}]
            if motor_Z is not None:
                motors += [{'name': 'motor_Z', 'values': format_nanarr(motor_Z), 'axis': [-3], 'unit': 'mm'}]
            ret_data['height'] = MetrologyData(height, unit='nm', pix_size=[x * 1e-3 for x in pix_size], pix_unit='mm',
                                               dim_detector=[-2, -1], axis_names=['Motor', 'Y', 'X'], motors=motors)
            if cam_size is not None:
                ret_data['height']._set_init_shape(cam_size)
        return ret_data

    def get_file_data(self, data, h5obj):
        ret_data = {}
        if type(h5obj) is h5py.File:
            ret_data = self.load_first_entry(data, h5obj)
        elif type(h5obj) is h5py.Group:
            if h5obj.attrs.get('NX_class') == b'NXwavefront':
                ret_data['height'] = data['unwrpdPhase']

        return ret_data

    def data_standard_format(self, data):
        """
        Convert data in standard format readable by orange pylost widgets, e.g. import heights / slopes_x as MetrologyData from the raw file data.

        :param data: Raw file data
        :type data: dict
        :return: Standard format data
        :rtype: dict
        """
        ret_data = {}
        try:
            h5path = data.get('h5path', '')
            with h5py.File(self.filename, 'r') as f:
                if h5path == '':
                    ret_data = self.load_first_entry(data, f)
                else:
                    if f[h5path].attrs.get('NX_class') == b'NXentry':
                        ret_data = {'module': 'scan_data', 'entry': h5path}
                        ret_data['scan_data'] = self.get_scans(data, f[h5path])
                    elif f[h5path].attrs.get('NX_class') == b'NXscan':
                        ret_data = self.get_scan(data, f[h5path])
                        ret_data['module'] = 'custom'
                        ret_data['h5path'] = h5path
                    else:
                        ret_data = self.get_file_data(data, f[h5path])
                        ret_data['module'] = 'custom'
                        ret_data['h5path'] = h5path

        except Exception as e:
            raise Exception('Error while converting to standard format: {}'.format(repr(e)))
        finally:
            return ret_data
