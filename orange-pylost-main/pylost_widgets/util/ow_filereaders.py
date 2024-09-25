import logging
import contextlib
import logging
import os
import pickle
from os import path, remove

import h5py
import numpy as np
import re
import silx
from Orange.data import FileFormat
from PyQt5 import uic
from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QDialogButtonBox, QHeaderView, QInputDialog, QAction
from astropy.units import Quantity
from silx.io.dictdump import h5todict, dicttoh5

from PyLOSt.data_in.fizeau.datx.read_datx import readDatxFile
from PyLOSt.data_in.esrf.ltp import LTPdata
from PyLOSt.data_in.esrf.veeco import OpdData
from PyLOSt.data_in.esrf.zygo import MetroProData, MxData
from PyLOSt.data_in.sharper.ImopHasoSlopes import ImopHasoSlopes
from pylost_widgets.util.DictionaryTree import DictionaryTreeDialog
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.resource_path import resource_path
from pylost_widgets.util.util_functions import format_nanarr, flip_data

log = logging.getLogger(__name__)

qtCreatorFile = resource_path(os.path.join("gui", "dialog_data_tree.ui"))  # Enter file here.
Ui_tree, QtBaseClass = uic.loadUiType(qtCreatorFile)
qtCreatorFile = resource_path(os.path.join("gui", "dialog_ascii_reader.ui"))  # Enter file here.
Ui_ascii, QtBaseClass = uic.loadUiType(qtCreatorFile)


class AsciiDialog(QDialog, Ui_ascii):
    """Dialog for loading ascii files"""

    def __init__(self, parent=None, filename=''):
        """
        Initialize Dialog for ascii file loader.

        :param parent: Parent object
        :type parent: QWidget
        :param filename: Loaded file name
        :type filename: str
        """
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.data = {}
        self.content = []
        self.buttonBox.button(QDialogButtonBox.Apply).setAutoDefault(False)
        self.buttonBox.button(QDialogButtonBox.Apply).setDefault(False)
        self.buttonBox.button(QDialogButtonBox.Cancel).setAutoDefault(False)
        self.buttonBox.button(QDialogButtonBox.Cancel).setDefault(False)
        self.buttonBox.button(QDialogButtonBox.Apply).clicked.connect(self.accept)
        if filename != '':
            with open(filename) as f:
                content = f.readlines()
            self.content = [x.strip() for x in content]
            self.skip_rows.returnPressed.connect(self.apply_options)
            self.delimiter.returnPressed.connect(self.apply_options)
            self.read_table_header.toggled.connect(self.apply_options)
            self.apply_options()

    def accept(self):
        """Callback after clicking apply button"""
        self.apply_options()
        super().accept()

    def apply_options(self):
        """Apply selected options (e.g. no of rows to skip) to the file data."""
        try:
            data = self.content
            skip = int(self.skip_rows.text())
            if skip > 0 and len(self.content) > skip:
                data = self.content[skip:]

            delim = self.delimiter.text()
            if delim in ['tab', '\\t']:
                delim = '\t'
            if delim != '':
                data = [[y.strip() for y in x.split(delim)] for x in data]

            keys = 'file_data'
            if self.read_table_header.isChecked() and len(data) >= 2:
                keys = data[0]
                data = data[1:]

            if any(data):
                try:
                    data = np.asarray(data, dtype=float)
                except Exception:
                    pass
                self.set_table_data(data)
                self.fill_data(data, keys)
                header = self.table.horizontalHeader()
                header.setSectionResizeMode(QHeaderView.ResizeToContents)

        except Exception as e:
            print('AsciiDialog-->apply_options')
            print(e)

    def set_table_data(self, content):
        """
        Display file lines in table (limited to 100 lines)

        :param content: File content
        :type content: list/np.ndarray
        """
        try:
            data = content[:100] if len(content) > 100 else content
            if isinstance(data, list):
                self.table.setRowCount(len(data))
                self.table.setColumnCount(1)
                for i, x in enumerate(data):
                    self.table.setItem(i, 0, QTableWidgetItem('{}'.format(x)))
            elif isinstance(data, np.ndarray) and data.ndim in [1, 2]:
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                self.table.setRowCount(data.shape[0])
                self.table.setColumnCount(data.shape[1])
                for i in np.arange(data.shape[0]):
                    for j in np.arange(data.shape[1]):
                        self.table.setItem(i, j, QTableWidgetItem('{}'.format(data[i, j])))
        except Exception as e:
            print('AsciiDialog-->set_table_data')
            print(e)

    def fill_data(self, data, keys):
        """
        Update data to retrieve in dictionary format. If header is selected, keys are taken from header names if they match the size of columns, else the header is used as a single key.

        :param data: File data formatted with current dialog options
        :type data: np.ndarray/list
        :param keys: Key names to convert to dictionary format, e.g. from header
        :type keys: list/str
        """
        try:
            self.data = {}
            if isinstance(data, np.ndarray) and data.ndim in [1, 2]:
                if data.ndim == 2 and isinstance(keys, (tuple, list, np.ndarray)) and len(keys) == data.shape[1]:
                    for i, key in enumerate(keys):
                        self.data[key] = data[:, i]
                elif isinstance(keys, (tuple, list, np.ndarray)):
                    self.data[keys[0]] = data
                elif isinstance(keys, str):
                    self.data[keys] = data
            else:
                key = keys[0] if isinstance(keys, list) else keys
                self.data = {key: data}
        except Exception as e:
            print('AsciiDialog-->fill_data')
            print(e)


class AsciiReader(FileFormat):
    """Base ascii file reader. Needs to be subclassed with field 'EXTENSIONS' to be used."""
    DESCRIPTION = 'Ascii file reader'
    SUPPORT_COMPRESSED = False
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 10

    def read(self):
        """
        Default reader for ascii file formats. It opens a dialog and provides options to skip rows, split by string etc., to format file data.

        :return: Formatted file data as dictionary
        :rtype: dict
        """
        data = {}
        dialog = AsciiDialog(None, self.filename)
        if dialog.exec_():
            data = dialog.data
        return data

    @classmethod
    def write_file(cls, filename, data):
        """
        Default ascii writer to save dictionary data.

        :param filename: Save file name
        :type filename: str
        :param data: Dictionary data
        :type data: dict
        """
        with open(filename, 'w') as file:
            for key in data:
                if isinstance(data[key], np.ndarray):
                    if isinstance(data[key], MetrologyData):
                        attrs = data[key].get_print_attributes()
                        file.write('\n{} attributes'.format(key))
                        for akey in attrs:
                            file.write('\n\t{} = {}'.format(akey, attrs[akey]))
                    file.write('\n{}\n'.format(key))
                    np.savetxt(file, data[key].view(np.ndarray), fmt='%.5f')
                else:
                    file.write('\n{} = {}'.format(key, data[key]))

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
            motors = []
            if 'slopes_x' in data and data['slopes_x'].ndim == 1:
                if 'x' in data:
                    data['x'] = data['x'] - np.nanmean(data['x'])
                    motors += [{'name': 'motor_X', 'values': format_nanarr(data['x']), 'axis': [-1],
                                'unit': data['xunit'] if 'xunit' in data else 'mm'}]
                    ret_data = {'slopes_x': MetrologyData(data['slopes_x'],
                                                          unit=data['unit'] if 'unit' in data else 'urad',
                                                          pix_size=1, axis_names=['Motor_x'], axis_values=['motor_X'],
                                                          motors=motors)}
        except Exception as e:
            raise Exception('Error while converting to standard format: {}'.format(repr(e)))
        finally:
            return ret_data


class TextDataReader(AsciiReader):
    """File loader subclassed from AsciiReader for txt, asc types. Uses default ascii reader and writer."""
    DESCRIPTION = 'Text data file reader'
    EXTENSIONS = ('.txt', '.asc')
    PRIORITY = 11


class H5TreeDialog(QDialog, Ui_tree):
    """Dialog for loading HDF5 files"""

    def __init__(self, parent=None, filename=''):
        """
        Initialization for HDF5 file loader dialog.

        :param parent: Parent object
        :type parent: QWidget
        :param filename: Loaded file name
        :type filename: str
        """
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.selected_item = ''
        self.selected_item_name = ''
        self.selected_item_type = None
        self.file_name = filename
        self.h5f = None

        self.__treeview = silx.gui.hdf5.Hdf5TreeView(self)
        self.model = self.__treeview.findHdf5TreeModel()
        self.__treeview.clicked.connect(self.selectItem)
        self.lt.addWidget(self.__treeview)
        if filename != '' and os.path.exists(filename):
            self.h5f = h5py.File(filename, 'a')
            self.model.clear()
            self.model.insertFile(filename)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.__treeview.addContextMenuCallback(self.customContextMenu)

    def accept(self):
        """Callback after clicking apply button"""
        try:
            self.h5f.close()
        except Exception:
            pass
        return super().accept()

    def reject(self):
        """Callback after clicking cancel button"""
        try:
            self.h5f.close()
        except Exception:
            pass
        return super().reject()

    def selectItem(self):
        """Callback after selecting an item in h5 tree"""
        selected = list(self.__treeview.selectedH5Nodes())
        if len(selected) >= 1:
            self.selected_item = selected[0].name
            self.selected_item_name = selected[0].basename
            self.selected_item_type = selected[0].ntype

    def customContextMenu(self, event):
        """Custom right click context for items in h5 tree, with options to add new group/sub-group"""
        selectedObjects = list(event.source().selectedH5Nodes())
        menu = event.menu()
        if len(selectedObjects) > 0:
            obj = selectedObjects[0]

            if obj.ntype is h5py.Group:
                action = QAction("Add new sub group", event.source())
                action.triggered.connect(lambda flag, h5obj=obj.h5py_object: self.create_new_group(flag, h5obj))
                menu.addAction(action)
            elif obj.ntype is h5py.File:
                action = QAction("Add new group", event.source())
                action.triggered.connect(lambda flag, h5obj=obj.h5py_object: self.create_new_group(flag, h5obj))
                menu.addAction(action)

    def create_new_group(self, flag, h5obj=None):
        """
        Create a new group in h5 file.

        :param flag: Action flag
        :type flag: bool
        :param h5obj: H5 parent node object
        :type h5obj: silx.gui.hdf5._utils.H5Node
        """
        if h5obj is None:
            raise Exception('No node is selected')
        text, ok = QInputDialog.getText(self, 'Add group', 'Group name: ')
        if ok:
            if text in h5obj:
                raise Exception('Group name already exists')
            else:
                self.h5f[h5obj.name].create_group(text)
            self.model.synchronizeIndex(self.model.index(0, 0, QModelIndex()))


class H5Reader(FileFormat):
    """File reader for hdf5 formats"""
    EXTENSIONS = ('.h5', '.hdf5', '.hdf')
    DESCRIPTION = 'HDF5 file reader'
    SUPPORT_COMPRESSED = False
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 2
    selected_group = ''
    open_dialog = True
    overwrite = True

    def read(self, dialog=None):
        """
        Reader for hdf5 file formats. It opens a dialog and provides option to load partially a single group in file.

        :return: Formatted file data as dictionary
        :rtype: dict
        """
        data = {}
        if self.open_dialog:
            self.selected_group = ''
            if dialog is None:
                dialog = H5TreeDialog(None, self.filename)
            if dialog.exec_():
                h5f = h5py.File(self.filename, 'r')
                if dialog.selected_item != '' and dialog.selected_item_type is not None:
                    if dialog.selected_item_type is h5py.Dataset:
                        item = dialog.selected_item_name if dialog.selected_item_name != '' else \
                            dialog.selected_item.split('/')[-1]
                        data[item] = h5f[dialog.selected_item][...]
                        data[item] = self.load_attributes_dataset(data[item], h5f[dialog.selected_item].attrs)
                    elif dialog.selected_item_type is h5py.Group:
                        self.selected_group = dialog.selected_item
                        data = self.dump_to_dict(h5f, path=dialog.selected_item)
                        self.load_attributes(data, h5f[dialog.selected_item])
                        data['h5path'] = dialog.selected_item
                    elif dialog.selected_item_type is h5py.File:
                        data = self.dump_to_dict(h5f)
                        self.load_attributes(data, h5f)
                else:
                    data = self.dump_to_dict(h5f)
                    self.load_attributes(data, h5f)
                h5f.close()
        else:
            h5f = h5py.File(self.filename, 'r')
            data = self.dump_to_dict(h5f, path=self.selected_group) if self.selected_group != '' else self.dump_to_dict(
                h5f)
            h5object = h5f[self.selected_group] if self.selected_group != '' else h5f
            self.load_attributes(data, h5object)
            h5f.close()

        return data

    def dump_to_dict(self,
                     h5f,
                     path="/",
                     exclude_names=None,
                     asarray=True,
                     dereference_links=True,
                     include_attributes=False,
                     errors='raise'):
        return h5todict(h5f, path, exclude_names, asarray, dereference_links, include_attributes, errors)

    @classmethod
    def write_file(cls, filename, data):
        """
        H5 writer to save dictionary data.

        :param filename: Save file name
        :type filename: str
        :param data: Dictionary data
        :type data: dict
        """
        if cls.selected_group != '':
            if cls.overwrite:
                h5f = h5py.File(filename, 'r+')
                if cls.selected_group in h5f:
                    del h5f[cls.selected_group]
                h5f.close()
            dicttoh5(data, filename, cls.selected_group, mode='r+', update_mode='replace')
        else:
            mode = 'w' if cls.overwrite else 'r+'
            dicttoh5(data, filename, mode=mode, update_mode='replace')
        # Save attributes
        h5f = h5py.File(filename, 'r+')
        try:
            h5object = h5f[cls.selected_group] if cls.selected_group != '' else h5f
            cls.update_attributes(data, h5object)
        except Exception as e:
            raise Exception('Error adding attributes to data')
        finally:
            h5f.close()

    @classmethod
    def update_attributes(cls, data, h5object):
        """
        Update attributes of hdf5 file datasets. MetrologyData or Quantity objects are saved as numpy arrays with their parameters (like units, pixel size) as hdf5 dataset attributes.
        Name of class is also saved as attribute.

        :param data: Dictionary data
        :type data: dict
        :param h5object: Selected h5 group/file object
        :type h5object:
        """
        for key in data:
            if isinstance(data[key], dict):
                cls.update_attributes(data[key], h5object[key])
            elif type(data[key]) == MetrologyData:
                h5object[key].attrs['class'] = 'MetrologyData'
                attributes = data[key].get_attributes()
                for akey in attributes:
                    h5object[key].attrs[akey] = attributes[akey]
            elif type(data[key]) == Quantity:
                h5object[key].attrs['class'] = 'Quantity'
                h5object[key].attrs['unit'] = '{}'.format(data[key].unit)

    def load_attributes(self, data, h5object):
        """
        Load hdf5 attributes and apply them.

        :param data: Output data after loading hdf5 file
        :type data: dict
        :param h5object: Selected h5 group/file object
        :type h5object:
        :return:
        :rtype:
        """
        for key in data:
            if isinstance(data[key], dict):
                data[key] = self.load_attributes(data[key], h5object[key])
            elif 'class' in h5object[key].attrs:
                data[key] = self.load_attributes_dataset(data[key], h5object[key].attrs)
            elif data[key].shape == ():
                data[key] = data[key].item()
        return data

    def load_attributes_dataset(self, data, attrs):
        """
        Load and apply hdf5 attributes for a dataset. Numpy arrays are converted to MetrologyData or Quantity if attribute 'class' with such value is present in dataset attributes.

        :param data: HDF5 dataset
        :type data: H5 compatible dataset types (np.ndarray, str, ...)
        :param attrs: Attributes of dataset
        :type attrs: dict
        :return: Converted dataset
        :rtype: MetrologyData/Quantity or same as 'data'
        """
        if 'class' in attrs:
            if attrs['class'] == 'Quantity' and 'unit' in attrs:
                data = Quantity(data, unit=attrs['unit'])
            elif attrs['class'] == 'MetrologyData' and 'unit' in attrs and 'pix_size' in attrs:
                data = MetrologyData.apply_attributes(data, dict(attrs))
        return data


class PickleReader(FileFormat):
    """File reader for pickle formats"""
    EXTENSIONS = ('.pkl',)
    DESCRIPTION = 'Pickle reader'
    SUPPORT_COMPRESSED = False
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 1

    def read(self):
        """
        Load pickle file

        :return: Loaded data
        :rtype: dict
        """
        data = pickle.load(open(self.filename, "rb"))
        dialog = DictionaryTreeDialog(parent=None, data=data)
        if dialog.exec_():
            return dialog.get_selected_data()
        return data

    @classmethod
    def write_file(cls, filename, data):
        """
        Wrtie dictionary data to pickle format.

        :param filename: Save file name
        :type filename: str
        :param data: Dictionary data to save
        :type data: dict
        """
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


class MetroProReader(FileFormat):
    """File reader for Zygo MetroPro/MX formats dat, datx"""
    EXTENSIONS = ('.dat', '.datx')
    DESCRIPTION = 'MetroPro file reader'
    SUPPORT_COMPRESSED = False
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 3
    PARAMS = {'instr_scale_factor': 1.0}
    clear_output_before_loading = True

    def read(self):
        """
        Load Zygo dat or datx file.

        :return: Loaded data
        :rtype: dict
        """
        data = {}
        if self.filename.endswith('.dat'):
            # data = readDatFile(self.filename)
            hgt = MetroProData.read(self.filename)
            data = hgt.__dict__
            if 'values' in data:
                data['values'] = np.moveaxis(data['values'], -1, -2)
            data = self._parse_comments(data, hgt.title)
        elif self.filename.endswith('.datx'):
            # data = readDatxFile(self.filename)
            hgt = MxData.read(self.filename)
            data = hgt.__dict__
            if 'values' in data:
                data['values'] = np.moveaxis(data['values'], -1, -2)
        return data

    def _parse_comments(self, data, comments):
        """
        Parse comments in dat file (implementation for ESRF Fizeau comment format), to extract motor X/Y positions.

        :param data: File data as dictionary
        :type data: dict
        :param comments: Comments string
        :type comments: str
        :return: Data dictionary with parser motor positions included
        :rtype: dict
        """
        try:
            # Current ESRF comments format 'Position= 1  X-coord= 32.75  X-shift= 0.929'
            cArr = re.split('[= ]', comments)
            cFilt = list(filter(bool, cArr))
            # if comments format changes??
            data['motorX'] = np.array(np.nan)
            data['motorY'] = np.array(np.nan)
            for i in np.arange(len(cArr) - 1):
                if cFilt[i] == 'X-coord':
                    data['motorX'] = np.asarray(-1 * np.double(cFilt[i + 1]))
                if cFilt[i] == 'X-shift':
                    data['motorXShift'] = np.asarray(-1 * np.double(cFilt[i + 1]))
                if cFilt[i] == 'Y-coord':
                    data['motorY'] = np.asarray(-1 * np.double(cFilt[i + 1]))
                if cFilt[i] == 'Y-shift':
                    data['motorYShift'] = np.asarray(-1 * np.double(cFilt[i + 1]))
        except Exception as e:
            print('_parse_comments <- MetroProReader')
            print(e)
        return data

    def get_start_pos_keys(self):
        """
        Start position locator keys. Used while loading multiple subaperture sequence to match data sizes.

        :return: [start pos y, start pos x]
        :rtype: [str, str]
        """
        return ['cn_org_y', 'cn_org_x']

    def get_stack_selected_keys(self):
        """
        Stack values in selected keys while loading file sequence.

        :return: list of keys which needs to be stacked, e.g. 'motorX'
        :rtype: list[str]
        """
        if self.filename.endswith('.datx'):
            return ['motorX', 'motorY', 'motorTz', 'motorRx', 'motorRy', 'motorRz']
        else:
            return []

    def get_cam_size(self):
        """
        Get keys locating camera size.

        :return: [camera height key, camera width key]
        :rtype: [str, str]
        """
        return ['ac_height', 'ac_width']

    @classmethod
    def write_file(cls, filename, data):
        """
        Wrtie dictionary data to Zygo dat format.

        :param filename: Save file name
        :type filename: str
        :param data: Dictionary data to save
        :type data: dict
        """
        pixel_size = None
        start_pos = None
        frame_size = None
        if isinstance(data, dict):
            if 'height' in data:
                data = data['height']
            else:
                raise Exception('Dictionary data has no key "height"')

        if isinstance(data, MetrologyData):
            pixel_size = data.pix_size_detector[0].to('m').value
            start_pos = data.start_position_pix
            frame_size = data.init_shape[::-1] if data.init_shape is not None and len(
                data.init_shape) == data.ndim else None
            data = data.to('m').value
        elif isinstance(data, np.ndarray):
            text, ok = QInputDialog.getText(None, cls.DESCRIPTION, 'Pixel size (um): ')
            if ok:
                pixel_size = float(text) * 1e-6
            data = data * 1e-9
        else:
            raise Exception('Data should be numpy array or MetrologyData')
        if data.ndim != 2:
            raise Exception('Data should have two dimensions')
        if pixel_size is None:
            raise Exception('Pixel size not found')
        MetroProData.writeMetroprofile(np.flip(data.T, axis=-1), pixel_size, filename, frame_size=frame_size)

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
            motors = []
            # if os.path.splitext(self.filename)[1].lower() == '.datx':
            if False:
                pix_sz = data['CameraRes'] * 1e3
                if 'motorX' in data:
                    motors += [{'name': 'motor_X', 'values': format_nanarr(data['motorX']), 'axis': [-3], 'unit': 'mm'}]
                if 'motorY' in data:
                    motors += [{'name': 'motor_Y', 'values': format_nanarr(data['motorY']), 'axis': [-3], 'unit': 'mm'}]
                height = MetrologyData(data['height_nm'], unit='nm', pix_size=data['CameraRes'] * 1e3,
                                       pix_unit='mm', dim_detector=[-2, -1], axis_names=['Motor', 'Y', 'X'],
                                       motors=motors)
                if data['ac_height'] != 0 and data['ac_width'] != 0:
                    height._set_init_shape([data['ac_height'], data['ac_width']])
                start_pos_pix = data['start_pos'] if 'start_pos' in data else [data['cn_org_y'], data['cn_org_x']]
            else:
                pix_sz = data['header']['lateral_res'] * 1e3
                if 'motorX' in data:
                    motors += [{'name': 'motor_X', 'values': format_nanarr(data['motorX']), 'axis': [-3], 'unit': 'mm'}]
                if 'motorY' in data:
                    motors += [{'name': 'motor_Y', 'values': format_nanarr(data['motorY']), 'axis': [-3], 'unit': 'mm'}]
                height = MetrologyData(data['values'],  # np.moveaxis(data['values'],-1,-2),
                                       unit='{}'.format(data['units']['values']),
                                       pix_size=data['header']['lateral_res'] * 1e3,
                                       pix_unit='mm', dim_detector=[-2, -1], axis_names=['Motor', 'Y', 'X'],
                                       motors=motors).to('nm')
                if data['header']['ac_height'] != 0 and data['header']['ac_width'] != 0:
                    height._set_init_shape([data['header']['ac_height'], data['header']['ac_width']])
                start_pos_pix = data['start_pos'] if 'start_pos' in data else [data['header']['cn_org_y'],
                                                                               data['header']['cn_org_x']]
            start_pos_pix = [Quantity(x) for x in start_pos_pix]
            height._set_start_position(start_pos_pix)
            try:
                # height.add_flag('invert_y_axis', True)
                height = flip_data(height, -2, flip_motors=['y'])
                # height = np.flip(height, -2)    # Flip data along Y axis
                # index_list = height.index_list
                # index_list[-2] = list(height.init_shape[-2] - index_list[-2]) if height.init_shape[-2] > np.max(index_list[-2]) else index_list[-2]
                # height._set_index_list(index_list)
            except Exception as e:
                raise Exception('Error while flipping Y axis')
            finally:
                ret_data = {'height': height}
            # self.PARAMS['pixel_size'] = [pix_sz*u.mm]*2
        except Exception as e:
            raise Exception('Error while converting to standard format: {}'.format(repr(e)))
        finally:
            return ret_data


class VeecoReader(FileFormat):
    """File reader for Veeco opd format"""
    EXTENSIONS = ('.OPD', '.opd',)
    DESCRIPTION = 'Veeco file reader'
    SUPPORT_COMPRESSED = False
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 3
    PARAMS = {'instr_scale_factor': 1.0}
    clear_output_before_loading = True

    def read(self):
        """
        Load Veeco MSI opd file.

        :return: Loaded data
        :rtype: dict
        """
        # data = Read_opd_file(self.filename)
        hgt = OpdData.read(self.filename)
        data = hgt.__dict__
        return data

    def get_stack_selected_keys(self):
        """
        Stack values in selected keys while loading file sequence.

        :return: list of keys which needs to be stacked, e.g. 'StageX'
        :rtype: list[str]
        """
        return ['StageX', 'StageY']

    @classmethod
    def write_file(cls, filename, data):
        """Not yet implemented"""
        pass

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
            # IN_TO_MM = 25.4
            motors = []
            if 'StageX' in data['header']:
                motors += [{'name': 'motor_X', 'values': format_nanarr(data['header']['StageX']), 'axis': [-3],
                            'unit': '{}'.format(data['units']['pixel'])}]
            if 'StageY' in data['header']:
                motors += [{'name': 'motor_Y', 'values': format_nanarr(data['header']['StageY']), 'axis': [-3],
                            'unit': '{}'.format(data['units']['pixel'])}]
            ret_data = {'height': MetrologyData(np.moveaxis(data['values'], -1, -2),
                                                unit='{}'.format(data['units']['values']),
                                                pix_size=data['header']['Pixel_size'],
                                                pix_unit='{}'.format(data['units']['pixel']),
                                                dim_detector=[-2, -1], axis_names=['Motor', 'Y', 'X'], motors=motors)
                        }
            # pix_sz = np.array(data['header']['Pixel_size'])
            # pix_unit = '{}'.format(data['units']['pixel'])
            # self.PARAMS['pixel_size'] = [Quantity(pix_sz, pix_unit)]*2
        except Exception as e:
            raise Exception('Error while converting to standard format: {}'.format(repr(e)))
        finally:
            return ret_data


class LTPReader(FileFormat):
    """File reader for ESRF LTP slp, slp2 formats"""
    EXTENSIONS = ('.slp2', '.slp')
    DESCRIPTION = 'LTP (ESRF) file reader'
    SUPPORT_COMPRESSED = False
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 3
    PARAMS = {'instr_scale_factor': 1.0}
    clear_output_before_loading = True

    def read(self):
        """
        Load LTP slp/slp2 file.

        :return: Loaded data
        :rtype: dict
        """
        # data = readSlp2File(self.filename)
        slp = LTPdata.read(self.filename)
        data = slp.__dict__
        try:
            if 'coords' in data and len(data['coords'] > 0):
                x = data['coords']
                data['start_pix'] = int(x[0] / np.min(np.diff(x)))
            # if 'start' in data and 'step' in data:
            #     data['start_pix'] = int(data['start']/data['step'])
        except Exception as e:
            print(e)
        return data

    def get_start_pos_keys(self):
        """
        Start position locator keys. Used while loading multiple subaperture sequence to match data sizes.

        :return: [start pos x]
        :rtype: [str]
        """
        return ['start_pix']

    def get_merge_selected_keys(self):
        """
        Merge values in selected keys while loading file sequence.

        :return: list of keys which needs to be merged, e.g. 'coords'
        :rtype: list[str]
        """
        return ['coords']

    @classmethod
    def write_file(cls, filename, data):
        """Not yet implemented"""
        pass

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
            motors = [{'name': 'motor_X', 'values': format_nanarr(data['coords']), 'axis': [-1],
                       'unit': '{}'.format(data['units']['coords'])}]
            ret_data = {'slopes_x': MetrologyData(data['values'], unit='{}'.format(data['units']['values']),
                                                  pix_size=1, axis_names=['Motor_x'], axis_values=['motor_X'],
                                                  motors=motors)
                        }
        except Exception as e:
            raise Exception('Error while converting to standard format: {}'.format(repr(e)))
        finally:
            return ret_data


class SharperReader(FileFormat):
    EXTENSIONS = ('.has',)
    DESCRIPTION = 'SHARPeR file reader'
    SUPPORT_COMPRESSED = False
    SUPPORT_SPARSE_DATA = False
    PRIORITY = 3
    PARAMS = {'instr_scale_factor': -0.5}
    clear_output_before_loading = True

    def read(self):
        data = {}
        if self.filename.endswith('.has'):
            obj = ImopHasoSlopes('Wrap', readXML=True, fname=self.filename)
            data = obj.__dict__
        return data

    def get_stack_selected_keys(self):
        """
        Stack values in selected keys while loading file sequence.

        :return: list of keys which needs to be stacked, e.g. 'motorX'
        :rtype: list[str]
        """
        return ['motorX', 'motorY', 'motorTz', 'motorRx', 'motorRy', 'motorRz']

    @classmethod
    def write_file(cls, filename, data):
        """Not yet implemented"""
        pass

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
            motors = []
            d = {'motorX': 'motor_X', 'motorY': 'motor_Y', 'motorTz': 'motor_Z', 'motorRx': 'motor_RX',
                 'motorRy': 'motor_RY', 'motorRz': 'motor_RZ'}
            du = {'motorX': 'mm', 'motorY': 'mm', 'motorTz': 'mm', 'motorRx': 'mrad', 'motorRy': 'mrad',
                  'motorRz': 'mrad'}
            for i, v in d.items():
                if i in data:
                    motors += [{'name': v, 'values': format_nanarr(data[i]), 'axis': [-3], 'unit': du[i]}]
            pix_sz = [data['steps'].Y / 1000, data['steps'].X / 1000]
            ret_data = {'slopes_x': MetrologyData(data['slopes_x'] * 1e3, unit='urad', pix_size=pix_sz, pix_unit='mm',
                                                  dim_detector=[-2, -1], axis_names=['Motors', 'Y', 'X'],
                                                  motors=motors),
                        'slopes_y': MetrologyData(data['slopes_y'] * 1e3, unit='urad', pix_size=pix_sz, pix_unit='mm',
                                                  dim_detector=[-2, -1], axis_names=['Motors', 'Y', 'X'], motors=motors)
                        }
            # self.PARAMS['pixel_size'] = [x*u.mm for x in pix_sz]
        except Exception as e:
            raise Exception('Error while converting to standard format: {}'.format(repr(e)))
        finally:
            return ret_data


# TODO: Need to correctly implement URL reader for h5 files and other formats
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse, unquote as urlunquote
from urllib.request import urlopen, Request
from pathlib import Path


class H5UrlReader(FileFormat):

    def __init__(self, filename):
        filename = filename.strip()
        if not urlparse(filename).scheme:
            filename = 'http://' + filename
        super().__init__(filename)

    @staticmethod
    def urlopen(url):
        req = Request(
            url,
            # Avoid 403 error with servers that dislike scrapers
            headers={'User-Agent': 'Mozilla/5.0 (X11; Linux) Gecko/20100101 Firefox/'})
        return urlopen(req, timeout=10)

    def read(self):
        self.filename = self._resolve_redirects(self.filename)

        with contextlib.closing(self.urlopen(self.filename)) as response:
            name = self._suggest_filename(response.headers['content-disposition'])
            # using Path since splitext does not extract more extensions
            extension = ''.join(Path(name).suffixes)  # get only file extension
            with NamedTemporaryFile(suffix=extension, delete=False) as f:
                f.write(response.read())
                # delete=False is a workaround for https://bugs.python.org/issue14243

            reader = self.get_reader(f.name)
            data = reader.read()
            remove(f.name)
        # Override name set in from_file() to avoid holding the temp prefix
        data.name = path.splitext(name)[0]
        data.origin = self.filename
        return data

    def get_extension(self):
        try:
            self.filename = self._resolve_redirects(self.filename)
            with contextlib.closing(self.urlopen(self.filename)) as response:
                name = self._suggest_filename(response.headers['content-disposition'])
                # using Path since splitext does not extract more extensions
                extension = ''.join(Path(name).suffixes)
                print('extension : ' + extension)
            return extension
        except Exception as ex:
            log.exception(ex)
            return None

    def _resolve_redirects(self, url):
        # Resolve (potential) redirects to a final URL
        with contextlib.closing(self.urlopen(url)) as response:
            return response.url

    def _suggest_filename(self, content_disposition):
        default_name = re.sub(r'[\\:/]', '_', urlparse(self.filename).path)

        # See https://tools.ietf.org/html/rfc6266#section-4.1
        matches = re.findall(r"filename\*?=(?:\"|.{0,10}?'[^']*')([^\"]+)",
                             content_disposition or '')
        return urlunquote(matches[-1]) if matches else default_name
