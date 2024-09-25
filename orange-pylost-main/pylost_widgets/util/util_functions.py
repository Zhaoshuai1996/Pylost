# coding=utf-8
import cProfile
import copy
import importlib
import os
import pkgutil
import pstats

import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QFileDialog, QMessageBox
from astropy import units as u
from astropy.units import Quantity
from line_profiler import LineProfiler
from orangewidget.utils.filedialogs import format_filter
from scipy.integrate import cumtrapz
from scipy.interpolate import interp2d
from silx.gui.colors import Colormap, registerLUT
from silx.io.dictdump import h5todict

from PyLOSt.algorithms.util.util_fit import evalPoly, fit_nD, getPixSz2D, getXYGrid
from PyLOSt.algorithms.util.util_integration_frankot_chellappa import frankot_chellappa
from PyLOSt.algorithms.util.util_integration_sylvester import g2s
from PyLOSt.databases.gs_table_classes import ConfigParams, Instruments, connectDB
from pylost_widgets.util.DictionaryTree import DictionaryTreeWidget
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.cm_scaling import ColorScale
from pylost_widgets.util.resource_path import resource_path

connectDB()

######################################################################################
## Common functions for all orange pylost widgets
MODULE_MULTI = ['scan_data']  # ['stitch_data', 'scan_data']
MODULE_SINGLE = ['custom']  # ['stitch_avg', 'scan_avg', 'custom']


def run_line_profiler(func, params):
    """
    Run line_profiler on given function and print statistics.

    :param func: Function to run
    :type func: Callable
    :param params: Parameters to function
    :type params: dict
    """
    lp = LineProfiler()
    lp_wrapper = lp(func)
    if params is not None:
        lp_wrapper(*params)
    lp.print_stats()


def start_profiler():
    """
    Start profiler for execution time analysis
    """
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler


def print_profiler(profiler):
    """
    Print captured profier statistics.

    :param profiler: Python profiler
    :type profiler: cProfile.Profile
    """
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()


def heights_to_metrologydata(scan, scan_vis, pix_size, x=None, y=None):
    """
    If slopes are in MetrologyData format, apply its attributes to integrated heights numpy array and convert to MetrologyData format.

    :param scan: Input scan datasets
    :type scan: dict
    :param scan_vis: Output scan datasets with heights data
    :type scan_vis: dict
    :param pix_size: Pixel size
    :type pix_size: list[Quantity]
    :param x: Axis x positions
    :type x: Quantity/np.ndarray
    :param y: Axis y positions
    :type y: Quantity/np.ndarray
    :return: height array
    :rtype: MetrologyData
    """
    if isinstance(scan['slopes_x'], MetrologyData) and scan_vis['height'] is not None:
        scan_vis['height'] = scan['slopes_x'].copy_to(scan_vis['height'])
        unit_x = pix_size[-1].unit if pix_size is not None and len(pix_size) > 0 and isinstance(pix_size[-1],
                                                                                                Quantity) else (
                1 * u.dimensionless_unscaled).unit
        unit_x = x.unit if x is not None and isinstance(x, Quantity) else unit_x
        scan_vis['height']._set_unit(
            scan['slopes_x'].unit.to('', equivalencies=u.dimensionless_angles()) * unit_x)
        scan_vis['height'] = scan_vis['height'].to('nm')
    return scan_vis['height']


def slopes_to_metrologydata(scan, scan_vis, pix_size, x=None, y=None):
    """
    If heights are in MetrologyData format, apply its attributes to differentiated slopes numpy arrays and convert to MetrologyData format.

    :param scan: Input scan datasets
    :type scan: dict
    :param scan_vis: Output scan datasets with slopes data
    :type scan_vis: dict
    :param pix_size: Pixel size
    :type pix_size: list[Quantity]
    :param x: Axis x positions
    :type x: Quantity/np.ndarray
    :param y: Axis y positions
    :type y: Quantity/np.ndarray
    :return: slopes x array, slopes y array
    :rtype: MetrologyData, MetrologyData
    """
    unit_x = pix_size[-1].unit if pix_size is not None and len(pix_size) > 0 and isinstance(pix_size[-1],
                                                                                            Quantity) else (
            1 * u.dimensionless_unscaled).unit
    unit_x = x.unit if x is not None and isinstance(x, Quantity) else unit_x
    if isinstance(scan['height'], MetrologyData) and scan_vis['slopes_x'] is not None:
        scan_vis['slopes_x'] = scan['height'].copy_to(scan_vis['slopes_x'])
        scan_vis['slopes_x']._set_unit(scan['height'].unit / unit_x)
        scan_vis['slopes_x'] = scan_vis['slopes_x'].to('urad', equivalencies=u.dimensionless_angles())
    if isinstance(scan['height'], MetrologyData) and scan_vis['slopes_y'] is not None:
        scan_vis['slopes_y'] = scan['height'].copy_to(scan_vis['slopes_y'])
        scan_vis['slopes_y']._set_unit(scan['height'].unit / unit_x)
        scan_vis['slopes_y'] = scan_vis['slopes_y'].to('urad', equivalencies=u.dimensionless_angles())
    return scan_vis['slopes_x'], scan_vis['slopes_y']


########################################################################################

def format_nanarr(arr):
    """
    If all elements of an array are nan's replace them with zeros.

    :param arr: Input array
    :type arr: np.ndarray
    :return: Formatted array
    :rtype: np.ndarray
    """
    arr = np.asarray(arr)
    try:
        if np.all(np.isnan(arr)):
            arr = np.asarray(np.nan_to_num(arr))
    except Exception as e:
        print(e)
    return arr


def plot_win_colormap(image_data, cmap_name='turbo', base='ra', factor=None):
    """
    Create and register a custom colormap based on statistical or fixed parameters.

    :param image_data: Image data
    :type image_data: MetrologyData/np.ndarray
    :param cmap_name: matplotlib colormap name
    :type cmap_name: str
    :param base: Base option in ['ra', 'rq', 'pv'].
    :type base: str
    :param factor: Factor used in rescaling colormap data
    :type factor: None
    :return: Colormap of type {name}_{base}
    :rtype: Colormap
    """
    try:
        if isinstance(image_data, MetrologyData):
            image_data = image_data.value
        image_data = image_data[~np.isnan(image_data)]
        if not np.isfinite(image_data).any():
            return Colormap(name='turbo')
        if factor is None:
            default = {'ra': 2.0, 'rq': 3.0, 'pv': 12.0}
            factor = default[base]
            if 'grey' in cmap_name.lower():
                factor = 4.7  # best for slopes
        lsc = ColorScale(image_data, cmap=cmap_name, name=cmap_name + '_' + base, base=base, zfac=factor)
        registerLUT(name=lsc.name, colors=lsc.cmap((np.linspace(0, 1, 256))))
        return Colormap(name=lsc.name, vmin=lsc.cmap_prms['zmin'], vmax=lsc.cmap_prms['zmax'])
    except Exception as e:
        print(e)
        return Colormap(name='turbo')


def copy_items(a, b, deepcopy=True, copydata=False):
    """
    Copy items from dictionary a to b. This does not create copy of arrays by default, lists or other python data. It only creates new references where possible.

    :param a: Source dictionary
    :type a: dict
    :param b: Destination dictionary
    :type b: dict
    :param deepcopy: Deepcopy recursively
    :type deepcopy: bool
    :param copydata: Create duplicate copies of numpy arrays with default data names (e.g slopes_x)
    :type copydata: bool
    """
    if deepcopy:
        for item in a:
            if type(a[item]) is dict:
                if item not in b:
                    b[item] = {}
                elif b[item] is not dict:
                    b[item] = {}
                copy_items(a[item], b[item], deepcopy=True)
            else:
                if copydata and item in list(DEFAULT_DATA_NAMES) + ['intensity']:
                    b[item] = np.empty_like(a[item])
                    np.copyto(b[item], a[item])
                else:
                    b[item] = a[item]
    else:
        for item in a:
            if copydata and item in list(DEFAULT_DATA_NAMES) + ['intensity']:
                b[item] = np.empty_like(a[item])
                np.copyto(b[item], a[item])
            else:
                b[item] = a[item]


def stack_dict(a, b, a_f=None, b_f=None, start_pos_keys=[], stack_selected_keys=[], cam_size_keys=[],
               merge_selected_keys=[], start_pos=None):
    """
    Stack dictionary objects (e.g. while loading sequence of files). Only numpy arrays, lists or tuples are stacked.

    :param a: Input dictionary
    :type a: dict
    :param b: Output stacked dictionary
    :type b: dict
    :param a_f: Parent input dictionary. By default parent is the same as input dictionary. If dict elements are sub directories, stack_dict is called iteratively, and a_f transfers link to parent dictionary
    :type a_f: dict
    :param b_f: Parent output stacked dictionary
    :type b_f: dict
    :param start_pos_keys: Start position keys in the dictionary. When stacking two images of different sizes from same instrument, start position of the image relative to detector is used for the merger.
    :type start_pos_keys: list[str]
    :param stack_selected_keys: Stack selected keys in the dictionary, even if the values are not numpy arrays or lists. E.g. stack motor position, even though it may be a float for a single image
    :type stack_selected_keys: list[str]
    :param cam_size_keys: Camera size keys in the dictionary. If provided and the images match the camera size, start_pos_keys are ignored.
    :type cam_size_keys: list[str]
    :param merge_selected_keys: Merge selected keys instead of stack into list, e.g. stitching LTP where a scan is a sequence of subscans with overlaps and the final list of positions is merger of subscan positions
    :type merge_selected_keys: list[str]
    :param start_pos: Updated start position of Output stacked arrays
    :type start_pos: list[float]
    :return: start positions keys, current start position
    :rtype: list[str], list[float]
    """
    if a_f is None:
        a_f = a
    if b_f is None:
        b_f = b
    for item in a:
        if type(a[item]) is dict:
            if item not in b:
                b[item] = {}
            elif type(b[item]) is not dict:
                b[item] = {}
            start_pos_keys, start_pos = stack_dict(a[item], b[item], a, b, start_pos_keys, stack_selected_keys,
                                                   cam_size_keys, merge_selected_keys, start_pos=start_pos)
        else:
            if isinstance(a[item], np.ndarray) and (item not in merge_selected_keys):
                start_pos_keys, start_pos = stack_nparrays(a, b, a_f, b_f, item, start_pos_keys,
                                                           cam_size_keys=cam_size_keys, start_pos=start_pos)

            if item in stack_selected_keys:
                a[item] = np.asarray(a[item])
                if item in b and type(b[item]) not in [np.ndarray, list]:
                    b[item] = np.asarray(b[item])
                start_pos_keys, start_pos = stack_nparrays(a, b, a_f, b_f, item, start_pos_keys,
                                                           cam_size_keys=cam_size_keys, start_pos=start_pos)
            elif item in merge_selected_keys and isinstance(a[item], (np.ndarray, tuple, list)):
                if item not in b:
                    b[item] = list(a[item])
                else:
                    b[item] = [x for x in b[item] if np.min(np.abs(x - a[item]) > 0.01)] + list(a[item])
            elif isinstance(a[item], list):
                if len(a[item]) == 1:
                    if item not in b:
                        b[item] = a[item]
                    else:
                        b[item] += a[item]
                else:
                    if item not in b:
                        b[item] = [a[item]]
                    else:
                        b[item] = list(b[item])
                        b[item].append([a[item]])
            else:
                if item not in b:
                    b[item] = a[item]
    return start_pos_keys, start_pos


def stack_nparrays(a, b, a_f, b_f, item, start_pos_keys, cam_size_keys=[], start_pos=None):
    """
    Stack numpy ndarrays. Output array b has shape (x,)+a.shape, where the new input a is attached at axis=0, to create final shape of (x+1,)+a.shape.

    :param a: Input dict
    :type a: dict
    :param b: Output stacked dict
    :type b: dict
    :param a_f: Parent input dictionary.
    :type a_f: dict
    :param b_f: Parent output stacked dictionary
    :type b_f: dict
    :param item: Selected key in the dictionary
    :type item: str
    :param start_pos_keys: Start position keys in the dictionary
    :type start_pos_keys: list[str]
    :param cam_size_keys: Camera size keys in the dictionary
    :type cam_size_keys: list[str]
    :param start_pos: Current start position of stacked output array
    :type start_pos: list[float]
    :return: start positions keys, current start position
    :rtype: list[str], list[float]
    """
    if item not in b:
        b[item] = a[item][np.newaxis]
    else:
        if isinstance(b[item], list):
            b[item] = b[item] + [a[item]]
        elif isinstance(b[item], np.ndarray):
            if np.all(np.isnan(b[item])) and not np.all(np.isnan(a[item])):
                b[item] = np.concatenate(
                    (np.full((b[item].shape[0],) + a[item].shape, np.nan, dtype=a[item].dtype), a[item][np.newaxis]),
                    axis=0)
            elif np.all(np.isnan(a[item])):
                b[item] = np.concatenate((b[item], np.full(b[item][0].shape, np.nan, dtype=a[item].dtype)[np.newaxis]),
                                         axis=0)
            else:
                if b[item].shape[1:] != a[item].shape:
                    if not any(start_pos_keys):
                        start_pos_keys = 'zeros'
                        #####################
                        ## Not working on another thread. Qt runs on main thread only
                        ## Need to either talk to main thread with signal/slots, or imlement start position after loading
                        # val = questionMsgAdv(title='Shape mismatch', msg='Shapes of "{}" do not match in file sequence. '
                        #                'Press "Yes" to select start position, "No" to pad NaNs at end.'.format(item))
                        # if val==2:
                        #     dialog = StartPosDialog(None, a_f, dim_pos=a[item].ndim)
                        #     if dialog.exec_():
                        #         if any(dialog.start_pos['pos']):
                        #             start_pos_keys = dialog.start_pos['pos']
                        # elif val==1:
                        #     start_pos_keys = 'zeros'
                if any(cam_size_keys) and b[item].shape[1:] == a[item].shape == tuple(
                        [get_dict_item(a_f, x) for x in cam_size_keys]):
                    b[item] = np.concatenate((b[item], a[item][np.newaxis]), axis=0)
                else:
                    if any(start_pos_keys):
                        if start_pos_keys == 'zeros':
                            pos_a = pos_b = [0] * a[item].ndim
                        elif start_pos_keys == 'met_start':
                            try:
                                pos_a = a[item].start_position_pix if isinstance(a[item], MetrologyData) else [0] * a[
                                    item].ndim
                                pos_b = b[item].start_position_pix if isinstance(b[item], MetrologyData) else [0] * a[
                                    item].ndim
                            except Exception as e:
                                print(e)
                                pos_a = pos_b = [0] * a[item].ndim
                        else:
                            pos_a = [get_dict_item(a_f, x) for x in start_pos_keys]
                            pos_b = b_f['start_pos'] if 'start_pos' in b_f else [get_dict_item(b_f, x) for x in
                                                                                 start_pos_keys]
                        b[item], start_pos = merge_arrays_at(a[item][np.newaxis], b[item], pos_a, pos_b)
                    else:
                        b[item] = np.concatenate((b[item], a[item][np.newaxis]), axis=0) if b[item].shape[1:] == a[
                            item].shape else list(b[item]) + [a[item]]

    return start_pos_keys, start_pos


def merge_arrays_at(a, b, st_a, st_b):
    """
    Merge two arrays at given start positions.

    :param a: Numpy array 1
    :type a: np.ndarray
    :param b: Numpy array 2
    :type b: np.ndarray
    :param st_a: Start position of array 1
    :type st_a: tuple
    :param st_b: Start position of array 2
    :type st_b: tuple
    :return: Merged array, start position of merged array
    :rtype: np.ndarray, tuple
    """
    try:
        st_a = np.asarray(st_a).flatten()
        st_b = np.asarray(st_b).flatten()
        en_a = [x + y for x, y in zip(st_a, a.shape[1:])]
        en_b = [x + y for x, y in zip(st_b, b.shape[1:])]
        pos_min = np.min(np.asarray([st_a, st_b]), axis=0)
        la = st_a - pos_min
        lb = st_b - pos_min
        pos_max = np.max(np.asarray([en_a, en_b]), axis=0)
        ra = pos_max - en_a
        rb = pos_max - en_b
        ac = np.pad(a, ((0, 0),) + tuple(zip(la, ra)), 'constant', constant_values=np.nan)
        bc = np.pad(b, ((0, 0),) + tuple(zip(lb, rb)), 'constant', constant_values=np.nan)
        return np.concatenate((bc, ac), axis=0), pos_min
    except Exception as e:
        print(e)
        return b, None


qtCreatorFile = resource_path(os.path.join("gui", "dialog_data_tree.ui"))  # Enter file here.
Ui_tree, QtBaseClass = uic.loadUiType(qtCreatorFile)


class StartPosDialog(QDialog, Ui_tree):
    def __init__(self, parent=None, data=None, dim_pos=2):
        """
        Start position key selection dialog, if no keys are provided by default,
        e.g. data = {
                        height = Array(m,n),
                        cn_org_x = 10,  <-- start position key along X
                        cn_org_y = 50,   <-- start position key along Y
                        intensity = Array(m,n),
                        ...
                    }

        :param parent: Parent objecy
        :type parent: QWidget
        :param data: Input dictionary data
        :type data: dict
        :param dim_pos: Number of keys or dimensions, default 2 for 2D images
        :type dim_pos: int
        """
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.start_pos = {'pos': []}
        self.data = data
        self.dim_pos = dim_pos
        self.header.setWordWrap(True)
        self.header.setText('Data : start_pos e.g.(Y,X)')
        self.setWindowTitle('Load start position')
        self.__treeViewer = DictionaryTreeWidget(self, None)
        self.__treeViewer.itemClicked.connect(self.itemClick)
        self.__treeViewer.updateDictionary(self.data)
        self.__treeViewer.setSelectionMode(QAbstractItemView.MultiSelection)
        self.layout.addWidget(self.__treeViewer)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def itemClick(self):
        """
        Callback from any click of the dictionary tree item. This method will update the start position keys.
        """
        select = self.__treeViewer.selectedItems()
        if len(select) > self.dim_pos:
            alertMsg('Selection', 'Please select only {} item(s)'.format(self.dim_pos))
            return
        self.start_pos = {'pos': []}
        pos = []
        for node in select:
            seldata = self.__treeViewer.get_selected_data(node)
            self.start_pos[node.text(0)] = seldata
            pos.append(node.text(0))
        self.start_pos['pos'] = pos
        if any(self.start_pos['pos']):
            self.header.setText('Data : start pos = {}'.format(self.start_pos))
        else:
            self.header.setText('Data : start_pos e.g.(Y,X)')


def has_key_in_dict(key, data):
    """
    Check if the given key is present in the dictionary or its sub dictionaries.

    :param key: Key to search
    :type key: str
    :param data: Input dictionary data
    :type data: dict
    :return: True if the key is present else False
    :rtype: bool
    """
    if key in data:
        return True
    for item in data:
        if type(data[item]) is dict:
            if has_key_in_dict(key, data[item]):
                return True
    return False


def walk_dict_by_type(dataset_type, data, path=[]):
    """
    Walk through dictionary and its subdictionaries, and return all the item paths of given datatype.

    :param dataset_type: Type of dataset to search
    :type dataset_type: class types
    :param data: Input dictionary data
    :type data: dict
    :param path: Path in the dictionary. It is passed in function parameters to use while iteration of subdictionaries
    :type path: list[str]
    :return: List of paths to items of given type
    :rtype: list[str]
    """
    retArr = []
    for item in data:
        if type(data[item]) is dict:
            path.append(item)
            out = walk_dict(dataset_type, data[item], path=path)
            if any(out):
                retArr += out
        elif type(data[item]) is dataset_type:
            retArr.append(path + [item])
    return retArr


def walk_dict(key, data, path=[]):
    """
    Walk through dictionary and its subdictionaries, and return all items with given key.

    :param key: Key to search
    :type key: str
    :param data: Input dictionary data
    :type data: dict
    :param path: Path in the dictionary. It is passed in function parameters to use while iteration of subdictionaries
    :type path: list[str]
    :return: List of paths to items of given key name
    :rtype: list[str]
    """
    retArr = []
    def_path = copy.deepcopy(path)
    for item in data:
        if item == key:
            retArr.append(path + [key])
        elif type(data[item]) is dict:
            path.append(item)
            out = walk_dict(key, data[item], path=path)
            if any(out):
                retArr += out
            else:
                path = copy.deepcopy(def_path)
    return retArr


def fill_dict(data, path, value):
    """
    Fill dictionary by given value at given path. Path is a list with first item linked to dictionray, second item linked to subdictionary and so on.

    Last item in the path is the actual key name
    :param data: Input dictionary data
    :type data: dict
    :param path: Path in list format
    :type path: list[str]
    :param value: Value to insert/replace
    :type value:
    """
    if any(data) and any(path):
        exec('data["{}"]=value'.format('"]["'.join(path)))


def get_dict_item(data, path):
    """
    Get dictionary item at given path. Path is a list with first item linked to dictionray, second item linked to subdictionary and so on
    Last item in the path is the actual key name.

    :param data: Input dictionary data
    :type data: dict
    :param path: Path in list format
    :type path: list[str]
    :return: Selected data
    :rtype:
    """
    if type(path) is not list:
        out = walk_dict(path, data, path=[])
        if any(out):
            path = out[0]
        else:
            return None
    if any(data) and any(path):
        data_sel = data
        for item in path:
            data_sel = data_sel[item]
        return data_sel
    else:
        return None


def get_import_data_names():
    """
    Get IMPORT_DATA_NAMES from the ConfigParams sql table. Import names are used in File loader widget, for a standard naming convention.
    e.g. slopes --> slopes_x, px --> pix_size.

    :return: List of import data names
    :rtype: list[str]
    """
    try:
        qnames = ConfigParams.selectBy(paramName='IMPORT_DATA_NAMES')[0]
        return qnames.paramValue.split(',')
    except:
        return []


def get_default_data_names():
    """
    Get DEFAULT_DATA_NAMES from the ConfigParams sql table. e.g. slopes_x, slopes_y, height.
    Default data names are processed or displayed in orange pylost widgets.

    :return: List of default data names
    :rtype: list[str]
    """
    try:
        qdef_names = ConfigParams.selectBy(paramName='DEFAULT_DATA_NAMES')[0]
        return qdef_names.paramValue.split(',')
    except:
        return []


DEFAULT_DATA_NAMES = get_default_data_names()


def get_setup_from_h5(h5Obj, setup):
    """
    Load stitching setup data from h5 object.

    :param h5Obj: H5 object
    :type h5Obj: H5Group
    :param setup: Stitching setup to load
    :type setup: str
    :return: dictionary of setup data
    :rtype: dict
    """
    if h5Obj is not None and setup in h5Obj:
        return h5todict(h5Obj.filename, setup)


def differentiate_heights(z, x=None, y=None, pix_sz=1, method='grad'):
    """
    Differentiate heights to get slopes, along last one (if curves) or two dimensions (if images).

    :param z: Height array (numpy ndarray or MetrologyData)
    :type z: np.ndarray / MetrologyData
    :param x: X positions
    :type x: np.ndarray
    :param y: Y positions
    :type y: np.ndarray
    :param pix_sz: Pixel size
    :type pix_sz: list[Quantity]
    :param method: Derivation method, default 'grad', options ['grad', 'diff']
    :type method: str
    :return: slopes_x (numpy ndarray or MetrologyData), slopes_y (numpy ndarray or MetrologyData)
    :rtype: MetrologyData, MetrologyData
    """
    pix_sz = getPixSz2D(pix_sz)
    sx = None
    sy = None
    if method == 'grad':
        dx = pix_sz[0] if x is None else x
        dy = pix_sz[1] if y is None else y
        if isinstance(dx, Quantity):
            sx = np.gradient(z, dx.value, axis=-1) / dx.unit
        else:
            sx = np.gradient(z, dx, axis=-1)
        if isinstance(dy, Quantity):
            sy = np.gradient(z, dy.value, axis=-2) / dy.unit if z.ndim >= 2 else None
        else:
            sy = np.gradient(z, dy, axis=-2) if z.ndim >= 2 else None
    elif method == 'diff':
        dx = pix_sz[0] if x is None else np.diff(x)
        dy = pix_sz[1] if y is None else np.diff(y)
        sx = np.divide(np.diff(z, axis=-1), dx)
        sy = np.divide(np.diff(z, axis=-2), dy) if z.ndim >= 2 else None

    if isinstance(z, MetrologyData):
        return sx * u.rad if sx is not None else sx, sy * u.rad if sy is not None else sy
    else:
        return sx, sy


def fit_nD_metrology(Z, pix_size=[1, 1], dtyp='height', **kwargs):
    """
    Fit numpy ndarray with polynomial, ellipse, etc, and remove the fit.

    :param Z: nD Data to fit
    :type Z: np.ndarray / MetrologyData
    :param pix_size: Pixel size
    :type pix_sz: list[Quantity]
    :param dtyp: Data type in ['height', 'slopes_x', 'slopes_y']
    :type dtyp: str
    :param kwargs: Additional arguments for the fit
    :type kwargs: dict
    :return: Coefficients of the fit, Fit residuals, Fit values
    :rtype: np.ndarray, MetrologyData, MetrologyData
    """
    if Z.ndim == 0:
        return None, Z, None
    dim_detector = [False] * Z.ndim
    dim_detector[-1] = True
    if Z.ndim > 1:
        dim_detector[-2] = True
    if isinstance(Z, MetrologyData):
        pix_size = Z.pix_size_detector
        pix_size = [x.value for x in pix_size]
        # Zm = np.nanmean(Z.flatten())
        ## Zscale = Zm.unit.si.scale
        # Zscale = Zm.unit.to('m') if dtyp=='height' else Zm.unit.to('rad')  # Assuming either heights or slopes
        Zscale = Z.unit.to('m') if dtyp == 'height' else Z.unit.to('rad')  # Assuming either heights or slopes
        Zval = Z.value
        dim_detector = Z.dim_detector
        axis_vals = Z.get_axis_val_items_detector()
        pix_scale = Z.get_pix_scale()
        axis_vals = [x.value for x in axis_vals]
    else:
        Zval = Z
        pix_scale = [1 for x in pix_size]
        Zscale = 1
        axis_vals = [[]] * Z.ndim

    Coeff_nD, Zn, Zn_fit = fit_nD(Zval, pix_size=pix_size, dim_detector=dim_detector, pix_scale=pix_scale,
                                  Zscale=Zscale, dtyp=dtyp, axis_vals=axis_vals, **kwargs)
    if isinstance(Z, MetrologyData):
        Zn = Z.copy_to(Zn)
        Zn_fit = Z.copy_to(Zn_fit)
    return Coeff_nD, Zn, Zn_fit


def integrate_slopes(sx, sy=None, x=None, y=None, pix_sz=1, isShapeRemoved=False, interpolate_nans=True, method=''):
    """
    Integrate slopes to heights. 1D integration of slopes_x, or 2D integration of slopes_x and slopes_y.

    :param sx: Slopes_x nd array
    :type sx: np.ndarray / MetrologyData
    :param sy: Slopes_y nd array
    :type sy: np.ndarray / MetrologyData
    :param x: X positions
    :type x: np.ndarray
    :param y: Y positions
    :type y: np.ndarray
    :param pix_sz: Pixel size
    :type pix_sz: list[Quantity]
    :param isShapeRemoved:  Is shape already removed (e.g. curvature for sperical/cylindrical mirror)
    :type isShapeRemoved: bool
    :param interpolate_nans: Interpolate nan values
    :type interpolate_nans: bool
    :param method: Integration method in ['trapz', 'cumtrapz'] for 1D curve integration, ['frankot_chellappa', 'sylvester'] for 2d integration
    :type method: str
    :return: Heights
    :rtype: np.ndarray / MetrologyData
    """
    pix_sz = getPixSz2D(pix_sz)
    if sx is None:
        raise Exception('Slopes_x object is not available.')
    if sy is None:
        return integrate_slopes_X(sx, x=x, pix_sz=pix_sz[0], method=method)

    if sx.ndim != sy.ndim:
        raise Exception('Dimensions of slopes_x and slopes_y do not match')

    return integrate_slopes_XY(sx, sy, x=x, y=y, pix_sz=pix_sz, isShapeRemoved=isShapeRemoved,
                               interpolate_nans=interpolate_nans, method=method)


def integrate_slopes_X(sx, x=None, pix_sz=1, method='trapz'):
    """
    Integrate slope curve(s) with 1D integration.

    :param sx: Slopes x nd array
    :type sx: np.ndarray / MetrologyData
    :param x: X positions
    :type x: np.ndarray
    :param pix_sz: Pixel size
    :type pix_sz: list[Quantity]
    :param method: Inegration method ['trapz', 'cumtrapz']
    :type method: str
    :return: Heights
    :rtype: np.ndarray / MetrologyData
    """
    z = None
    if method in ['trapz', 'cumtrapz']:
        z = cumtrapz(sx, x=x, dx=pix_sz, axis=-1, initial=0)  # uses dx if x is None, integrate along last dimention
    return z


def integrate_slopes_XY(sx, sy, x=None, y=None, pix_sz=[1, 1], isShapeRemoved=False, interpolate_nans=True,
                        method='frankot_chellappa'):
    """
    Integrate slopes_x, slopes_y images to heights with 2D integration. Remove polynomial degree 1 (i.e. curvature), before integration if not already removed.

    :param sx: Slopes x nd array
    :type sx: np.ndarray / MetrologyData
    :param sy: Slopes y nd array
    :type sy: np.ndarray / MetrologyData
    :param x: X positions
    :type x: np.ndarray
    :param y: Y positions
    :type y: np.ndarray
    :param pix_sz: Pixel size
    :type pix_sz: list[Quantity]
    :param isShapeRemoved:  Is shape already removed (e.g. curvature for sperical/cylindrical mirror)
    :type isShapeRemoved: bool
    :param interpolate_nans: Interpolate nan values
    :type interpolate_nans: bool
    :param method: Inegration method ['frankot_chellappa', 'sylvester']
    :type method: str
    :return: Heights
    :rtype: np.ndarray / MetrologyData
    """
    z = None
    # integrate only residuals of plane fit, ideally shape should be removed e.g. ellipse
    sx_resd = sx
    sy_resd = sy
    if not isShapeRemoved:
        coef_x, sx_resd, _ = fit_nD_metrology(sx, pix_size=pix_sz, degree=1, dtyp='slopes_x')
        coef_y, sy_resd, _ = fit_nD_metrology(sy, pix_size=pix_sz, degree=1, dtyp='slopes_y')

    if sx.ndim > 2:
        z = np.full_like(sx * pix_sz[0], np.nan)
        if isinstance(z, MetrologyData):
            z = z.to('nm', equivalencies=u.dimensionless_angles())
        for idx in np.ndindex(sx.shape[:-2]):
            z[idx] = integrate_slopes_XY_2D(sx_resd[idx], sy_resd[idx], x=x, y=y, pix_sz=pix_sz,
                                            interpolate_nans=interpolate_nans, method=method)
    elif sx.ndim == 2:
        z = integrate_slopes_XY_2D(sx_resd, sy_resd, x=x, y=y, pix_sz=pix_sz, interpolate_nans=interpolate_nans,
                                   method=method)
    else:
        z = integrate_slopes_XY_2D(sx_resd[np.newaxis], sy_resd[np.newaxis], x=x, y=y, pix_sz=pix_sz,
                                   interpolate_nans=interpolate_nans, method=method)

    # TODO: add shape to z
    if not isShapeRemoved:
        cz = get_coef_z(coef_x, coef_y)
        xv, yv = getXYGrid(z, pix_sz, order=2, mask=~np.isnan(z))
        if isinstance(sx, MetrologyData):
            scale = sx.unit.to('rad') * xv.unit.to('m') / z.unit.to('m')
            xv = xv.value
            yv = yv.value
            Zfit = scale * evalPoly(cz, xv, yv, 2, 0, nbVar=2, terms=[], dtyp='height') * z.unit
        else:
            Zfit = evalPoly(cz, xv, yv, 2, 0, nbVar=2, terms=[], dtyp='height')
        z = z + Zfit
    return z


def get_coef_z(cx, cy):
    """
    Get polynomial fit coefficients of heights from fit coefficients of slopes.

    :param cx: Fit coefficients of slopes_x
    :type cx: np.ndarray
    :param cy: Fit coefficients of slopes_y
    :type cy: np.ndarray
    :return: Fit coefficients of heights
    :rtype: np.ndarray
    """
    # cx = l + my + nx, cy = o + py + qx
    # cz = 0 + oy + lx + p/2 y**2 + (m+q) xy + n/2 x**2
    cx = np.array(cx)
    cy = np.array(cy)
    cz = np.zeros((*cx.shape[:-1], 6), dtype=float)
    cz[..., 0] = 0
    cz[..., 1] = cy[..., 0]
    cz[..., 2] = cx[..., 0]
    cz[..., 3] = cy[..., 1] / 2
    cz[..., 4] = cx[..., 1] + cy[..., 2]
    cz[..., 5] = cx[..., 2] / 2
    return cz


def integrate_slopes_XY_2D(sx, sy, x=None, y=None, pix_sz=[1, 1], interpolate_nans=True, method='frankot_chellappa'):
    """
    Integrate slopes_x, slopes_y images to heights with 2D integration.

    :param sx: Slopes x nd array
    :type sx: np.ndarray / MetrologyData
    :param sy: Slopes y nd array
    :type sy: np.ndarray / MetrologyData
    :param x: X positions
    :type x: np.ndarray
    :param y: Y positions
    :type y: np.ndarray
    :param pix_sz: Pixel size
    :type pix_sz: list[Quantity]
    :param interpolate_nans: Interpolate nans
    :type interpolate_nans: bool
    :param method: Integration method in ['frankot_chellappa', 'sylvester']
    :type method: str
    :return: Heights
    :rtype: np.ndarray / Quantity
    """
    z = np.full_like((sx.value if isinstance(sx, Quantity) else sx), np.nan)
    msk = ~np.isnan(sx) & ~np.isnan(sy)
    if not msk.any():
        return z
    v = np.transpose(np.nonzero(msk))
    mn = np.min(v, axis=0)
    mx = np.max(v, axis=0)
    m = (slice(mn[0], mx[0] + 1), slice(mn[1], mx[1] + 1))  # np.ix_(msk.any(axis=1),msk.any(axis=0))
    if isinstance(pix_sz[0], Quantity) and isinstance(pix_sz[1], Quantity):
        pix_unit = [x.unit for x in pix_sz]
        pix_sz = [x.to(pix_unit[0]).value for x in pix_sz]
    sx_m = sx.value[m] if isinstance(sx, Quantity) else sx[m]
    sy_m = sy.value[m] if isinstance(sy, Quantity) else sy[m]
    if (not np.any(sx_m)) or (not np.any(sy_m)):
        return z

    (ny, nx) = sx.shape
    if x is None:
        x = np.linspace(0, nx - 1, num=nx) * pix_sz[0]
    if y is None:
        y = np.linspace(0, ny - 1, num=ny) * pix_sz[1]
    x_m = x[m[1]]
    y_m = y[m[0]]
    if interpolate_nans:
        xx, yy = np.meshgrid(x_m, y_m)
        sx_nans = np.isnan(sx_m)
        sy_nans = np.isnan(sy_m)
        if sx_nans.any():
            f = interp2d(xx[~sx_nans], yy[~sx_nans], sx_m[~sx_nans])
            sx_m = f(x_m, y_m)
        if sy_nans.any():
            f = interp2d(xx[~sy_nans], yy[~sy_nans], sy_m[~sy_nans])
            sy_m = f(x_m, y_m)

    if method == 'sylvester':
        z[m] = g2s(x_m, y_m, sx_m, sy_m)
    elif method == 'frankot_chellappa':
        z[m] = - 0.5 * (pix_sz[0] + pix_sz[1]) * frankot_chellappa(sx_m, sy_m, reflec_pad=True)

    if isinstance(sx, Quantity):
        z = (z * pix_unit[0] * sx.unit).to('nm', equivalencies=u.dimensionless_angles())
    return z


def compare_shapes(shp1, shp2):
    """
    Compare two shapes, returns True if all dimension sizes match. Dimensions with size 1 or none are ignored.

    :param shp1: Shape 1
    :type shp1: tuple
    :param shp2: Shape 2
    :type shp2: tuple
    :return: Boolean whether the shapes match
    :rtype: bool
    """
    shp1 = tuple(shp1)
    shp2 = tuple(shp2)
    if len(shp1) > len(shp2):
        shp2 = (1,) * (len(shp1) - len(shp2)) + shp2
    elif len(shp1) < len(shp2):
        shp1 = (1,) * (len(shp2) - len(shp1)) + shp1

    for i, val in enumerate(shp1):
        if val == 1:
            continue
        elif shp2[i] == 1:
            continue
        elif val == shp2[i]:
            continue
        else:
            return False
    return True


def match_size(scan, scan_second, scan_item, second_item):
    """
    Match sizes of two items (numpy arrays) and if they match return the second item. Else apply mask of start & end positions to second item and return it.

    :param scan: Scan dictionary
    :type scan: dict
    :param scan_second: Second scan dictionary
    :type scan_second: dict
    :param scan_item: Scan item
    :type scan_item: np.ndarray
    :param second_item: Second scan item
    :type second_item: np.ndarray
    :return: Processed second item
    :rtype: np.ndarray
    """
    if compare_shapes(second_item.shape, scan_item.shape):
        return second_item
    else:
        if 'full_size' in scan and 'start_pos' in scan and 'end_pos' in scan:
            if questionMsg(title='Shape mismatch',
                           msg='Shapes mismatch. First dataset has mask applied. Click "Yes" to check if full shapes before mask match.'):
                shape_full = scan['full_size']
                st = scan['start_pos']
                en = scan['end_pos']
                shape_second = scan_second['full_size'] if 'full_size' in scan_second else second_item.shape
                st_2 = scan_second['start_pos'] if 'start_pos' in scan_second else [0] * len(st)
                en_2 = scan_second['end_pos'] if 'end_pos' in scan_second else [x + y for x, y in zip(st_2, second_item.shape[-2:] if second_item.ndim > 1 else second_item.shape)]
                if compare_shapes(shape_full, shape_second):
                    if scan_item.ndim == 1 and second_item.ndim == 1 and len(st) == 1 and len(en) == 1:
                        if st[0] < st_2[0] or en[0] > en_2[0]:
                            raise Exception(
                                'Cannot apply mask of first data to second data. First data mask is outside the bounds of second data mask')
                        if en[0] - st[0] + 1 != scan_item.shape[-1]:
                            raise Exception('Mask start & end positions are not up to date with the scan data')
                        return second_item[st[0] - st_2[0]:en[0] - st_2[0] + 1]
                    elif scan_item.ndim >= 2 and second_item.ndim >= 2 and len(st) == 2 and len(en) == 2:
                        if st[0] < st_2[0] or en[0] > en_2[0] or st[1] < st_2[1] or en[1] > en_2[1]:
                            raise Exception(
                                'Cannot apply mask of first data to second data. First data mask is outside the bounds of second data mask')
                        if en[0] - st[0] + 1 != scan_item.shape[-2] and en[1] - st[1] + 1 != scan_item.shape[-1]:
                            raise Exception('Mask start & end positions are not up to date with the scan data')
                        slc = []
                        if second_item.ndim > 2:
                            slc = [slice(None)] * (second_item.ndim - 2)
                        slc += [slice(st[0] - st_2[0], en[0] - st_2[0] + 1),
                                slice(st[1] - st_2[1], en[1] - st_2[1] + 1)]
                        second_item_mask = second_item[tuple(slc)]
                        return second_item_mask
                    else:
                        raise Exception('Shape mismatch between datasets')
                else:
                    raise Exception(
                        'Trying to apply mask data from first dataset. Full shape (before any mask) mismatch between two datasets')
        else:
            raise Exception('Shape mismatch and no mask available in (first) scan data')


def get_mean_image(data):
    """
    If number of data dimensions are >2, it is averaged along all other except last two.

    :param data: Input data numpy nd array
    :type data: np.ndarray
    :return: Mean data, Mean data in original data shape
    :rtype: np.ndarray, np.ndarray
    """
    if data.ndim <= 2:
        return data, data
    else:
        slc = [np.newaxis] * (data.ndim - 2)
        for i in range(0, data.ndim - 2):
            data = np.nanmean(data, axis=0)
        data_dim = data[tuple(slc)]
        return data, data_dim


def get_data_from_h5(h5Obj, entry, data_obj):
    """
    Get data from h5 object with given entry name.

    :param h5Obj: H5 object
    :type h5Obj: H5Group
    :param entry: Entry name
    :type entry: str
    :param data_obj: Input dictionary data reference
    :type data_obj: dict
    """
    scans = {}
    if h5Obj is not None:
        h5msr = h5Obj[entry]
        details = get_instrument_details(h5msr)
        if details is not None:
            data_obj['instrument_id'] = details.instrId
        pix_size = np.double(h5msr['Instrument/resolution'][...])
        pix_size_unit = h5msr['Instrument/resolution'].attrs['units']
        data_obj['instr_scale_factor'] = float(
            h5msr['Instrument/scale_factor'][...]) if 'Instrument/scale_factor' in h5msr else 1.0
        # Scan data
        h5scans = h5msr['Data']
        for it in h5scans.keys():
            if 'NX_class' in h5scans[it].attrs and h5scans[it].attrs['NX_class'] == 'NXdata':  # loop over scans
                h5scan = h5scans[it]
                scan = {}
                mask = None if 'mask' not in h5scan else h5scan['mask'][...]
                has_data = False
                motors = []
                for i in ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']:
                    motor_i = 'motor_{}'.format(i)
                    if motor_i in h5scan:
                        motor_arr = h5scan[motor_i][...]
                        motor_unit = h5scan[motor_i].attrs['units'] if 'units' in h5scan[motor_i].attrs else ''
                        # scan[motor_i] = Quantity(motor_arr, unit=motor_unit)
                        if not np.all(np.isnan(motor_arr)):
                            m = {'name': motor_i, 'values': motor_arr, 'axis': [-3], 'unit': '{}'.format(motor_unit)}
                            motors.append(m)
                for item in get_default_data_names():
                    if item in h5scan:
                        item_units = h5scan[item].attrs['units'] if 'units' in h5scan[item].attrs else ''
                        item_scale = float(h5scan[item].attrs['scale']) if 'scale' in h5scan[item].attrs else 1.0
                        item_data = MetrologyData(h5scan[item][...] * item_scale, unit=item_units,
                                                  pix_size=pix_size, pix_unit=pix_size_unit,
                                                  dim_detector=[-2, -1], axis_names=['Motor', 'Y', 'X'],
                                                  motors=motors)
                        scan[item], _, _ = apply_mask(item_data, mask)
                        has_data = True
                if not has_data:
                    raise Exception('Scan {} has no primary data. e.g. slopes_x, height'.format(it))

                scans[it] = scan
    data_obj['scan_data'] = scans


def get_params_data(data_obj):
    """
    Get number of subapertures.

    :param data_obj: Input data dictionary
    :type data_obj: dict
    :return: Number of subapertures
    :rtype: int
    """
    nb_subaps = 0
    pix_sz = None
    mXArr = np.array([])
    mYArr = np.array([])
    if 'scan_data' in data_obj:
        scan = data_obj['scan_data'][data_obj['scan_data'].keys()[0]]
        nb_subaps, pix_sz, mXArr, mYArr = get_params_scan(scan)
    elif any(set(DEFAULT_DATA_NAMES).intersection(data_obj.keys())):
        nb_subaps, pix_sz, mXArr, mYArr = get_params_scan(data_obj)
    return nb_subaps, pix_sz, mXArr, mYArr


def get_stitch_step(data_obj, mXArr=None, mYArr=None):
    """
    Get stitching step from a single or sequence of scans.

    :param data_obj: Input data dictionary
    :type data_obj: dict
    :return: Stitch step X, Stitch step Y
    :rtype: float, float
    """
    step_x = 0.0
    step_y = 0.0
    try:
        if not (np.any(mXArr) and np.any(mYArr)):
            _, _, mXArr, mYArr = get_params_data(data_obj)

        if len(mXArr) > 0:
            step_x = np.mean(np.diff(mXArr))
            if isinstance(step_x, Quantity):
                step_x = step_x.to('mm').value

        if len(mYArr) > 0:
            step_y = np.mean(np.diff(mYArr))
            if isinstance(step_y, Quantity):
                step_y = step_y.to('mm').value
    except Exception as e:
        step_x = step_y = 0.0
        print(e)
    return step_x, step_y


def update_motors(scan_item, scan_out_item, mask_list):
    """
    Update motor positions from input scan item to output scan item.

    :param scan_item: Scan item (e.g height)
    :type scan_item: MetrologyData
    :param scan_out_item: Output scan item
    :type scan_out_item: MetrologyData
    :param mask_list: Mask on subapertures
    :type mask_list: list
    """
    try:
        if any(scan_item._motors):
            for i, val in enumerate(scan_item.dim_detector):
                for j in range(len(scan_item._motors)):
                    m = scan_item._motors[j]
                    if not val and 'values' in m and scan_item.shape[i] == len(m['values']):
                        scan_out_item._motors[j]['values'] = m['values'][mask_list[i]]

    except Exception as e:
        print(e)


def load_axis_pix(data_obj, step_x, step_y, use_step_always=False):
    """
    Load axis values in pixels from a scan or a sequence of scans.

    :param data_obj: Input data dictionary
    :type data_obj: dict
    :param step_x: Step size X
    :type step_x: float
    :param step_y: Step size Y
    :type step_y: float
    :param use_step_always: Use stitch step instead of motor values
    :type use_step_always: bool
    :return: Updated scans
    :rtype: dict
    """
    if 'scan_data' in data_obj:
        pix_sz = None
        if 'pix_size' in data_obj:
            if isinstance(data_obj['pix_size'], (list, np.ndarray, tuple)) \
                    and np.all([isinstance(x, Quantity) for x in data_obj['pix_size']]):
                pix_sz = data_obj['pix_size']
        for key in data_obj['scan_data']:
            scan = data_obj['scan_data'][key]
            if type(scan) is not dict:
                continue
            n_sub, pix_sz, mXArr, mYArr = get_params_scan(scan, pix_sz=pix_sz)
            if pix_sz is None:
                raise Exception('Pixel size not found')
            mxp, myp = get_axis_pixels(mXArr, mYArr, pix_sz, step_x, step_y, n_sub=n_sub,
                                       use_step_always=use_step_always)
            scan['motor_X_pix'] = mxp
            if myp is not None:
                scan['motor_Y_pix'] = myp
    elif any(set(DEFAULT_DATA_NAMES).intersection(data_obj.keys())):
        n_sub, pix_sz, mXArr, mYArr = get_params_scan(data_obj)
        if pix_sz is None:
            raise Exception('Pixel size not found')
        mxp, myp = get_axis_pixels(mXArr, mYArr, pix_sz, step_x, step_y, n_sub=n_sub, use_step_always=use_step_always)
        data_obj['motor_X_pix'] = mxp
        if myp is not None:
            data_obj['motor_Y_pix'] = myp
    return data_obj


def get_params_scan(scan, pix_sz=None):
    """
    Get parameters from scan dictionary.

    :param scan: Scan dictionary
    :type scan: dict
    :param pix_sz: Pixel size
    :type pix_sz: list[Quantity]
    :return: Number of subapertures, pixel size, motor x positions, motor y positions
    :rtype: int, list[Quantity], Quantity/np.ndarray, Quantity/np.ndarray
    """
    keys = list(set(DEFAULT_DATA_NAMES).intersection(scan.keys()))
    obj = scan[keys[0]]
    nb_subapertures = obj.shape[0]
    mXArr = scan['motor_X'] if 'motor_X' in scan else []
    mYArr = scan['motor_Y'] if 'motor_Y' in scan else []
    if type(obj) is MetrologyData and any(obj.motors):
        for d in obj.motors:
            mXArr = Quantity(d['values'], unit=d['unit']) if d['name'] == 'motor_X' else mXArr
            mYArr = Quantity(d['values'], unit=d['unit']) if d['name'] == 'motor_Y' else mYArr

    if type(obj) is MetrologyData:
        pix_sz = obj.pix_size_detector
    elif 'pix_size' in scan:
        pix_sz = scan['pix_size']

    return nb_subapertures, pix_sz, mXArr, mYArr


def get_axis_pixels(mXArr, mYArr, pix_sz, step_x, step_y, n_sub=0, use_step_always=False):
    """
    Get X,Y axes position values in pixels.

    :param mXArr: X positions
    :type mXArr: Quantity / np.ndarray
    :param mYArr: Y positions
    :type mYArr: Quantity / np.ndarray
    :param pix_sz: Pixel size
    :type pix_sz: list[Quantity]
    :param step_x: Stitching step along X axis
    :type step_x: float
    :param step_y: Stitching step along Y axis
    :type step_y: float
    :param n_sub: Number of subapertures
    :type n_sub: int
    :param use_step_always: Always use stitch steps instead of mXArr, mYArr
    :type use_step_always: bool
    :return: X positions in pixels, Y positions in pixels
    :rtype: np.ndarray, np.ndarray
    """
    has_x = mXArr is not None and len(mXArr) > 0 and not np.isnan(np.sum(mXArr))
    has_y = mYArr is not None and len(mYArr) > 0 and not np.isnan(np.sum(mYArr))
    try:
        if np.all(mXArr == 0.0) and np.all(mYArr == 0.0):
            has_x = False
            has_y = False
    except:
        pass
    if use_step_always:
        has_x = False
        has_y = False
        if mXArr is not None and len(mXArr) > 0:
            mXArr[:] = 0
        if mYArr is not None and len(mYArr) > 0:
            mYArr[:] = 0

    if not has_x and step_x != 0:
        has_x = True
        mXArr = Quantity(np.arange(0, n_sub) * step_x, unit='mm')
    if not has_y and step_y != 0:
        has_y = True
        mYArr = Quantity(np.arange(0, n_sub) * step_y, unit='mm')
    if not has_x and not has_y:
        raise Exception('Please enter stitch step x or y.')

    mxp = arr_to_pix(mXArr, pix_sz[-1], n_sub)[0]
    myp = arr_to_pix(mYArr, pix_sz[-2 if len(pix_sz) > 1 else -1], n_sub)[0]
    return mxp, myp


def arr_to_pix(a, p, n_sub=0):
    """
    Array (e.g. motor X) converted to pixel format.

    :param a: Input array
    :type a: Quantity / np.ndarray
    :param p: Pixel size
    :type p: Quantity / float
    :param n_sub: Number of subapertures in sequence
    :type n_sub: int
    :return: Pixel format array, residual pixel offsets
    :rtype: np.ndarray[int], np.ndarray[float]
    """
    if a is None:
        return np.zeros(n_sub, dtype=int) if n_sub > 0 else None, None
    if len(a) == 0:
        return np.zeros(n_sub, dtype=int) if n_sub > 0 else None, None
    if np.sum(a) in [0, np.nan, np.inf]:
        return np.zeros(a.shape if isinstance(a, np.ndarray) else len(a), dtype=int), np.zeros(
            a.shape if isinstance(a, np.ndarray) else len(a), dtype=int)

    ar = a - np.min(a)
    if type(a) is Quantity and isinstance(p, Quantity):
        div = (ar / p).to('').value
    else:
        # ar = ar.value if type(a) is Quantity else ar
        # p = p.value if type(p) is Quantity else p
        div = (ar / p)
    ap = np.rint(div)
    da = div - ap
    return ap.astype(int), da  # pixel offsets from start


def get_pix_size(Z=None, scan={}):
    """
    Get pixel size of dataset

    :param Z: Input dataset
    :type Z: MetrologyData/np.ndarray
    :param scan: Scan dictionary with pixel size, if dataset is only numpy array and not MetrologyData
    :type scan: dict
    :return: Pixel size
    :rtype: list[Quantity]
    """
    pix_sz = None
    try:
        pix_sz = scan['pix_size'] if 'pix_size' in scan else None
        if Z is not None and isinstance(Z, MetrologyData):
            pix_sz = Z.pix_size_detector
    except Exception as e:
        print(e)
    return pix_sz


def get_dims(Z):
    """
    Get detector dimensions of an array.

    :param Z: Input array
    :type Z: MetrologyData/np.ndarray
    :return: Detector dimensions
    :rtype: np.ndarray[bool]
    """
    dims = [False] * Z.ndim
    dims[-1] = True
    if Z.ndim > 1:
        dims[-2] = True
    if isinstance(Z, MetrologyData):
        dims = Z.dim_detector
    dims = np.array(dims)
    return dims


def remove_tuple(d1, d2):
    """Subtract second tuple from first and return subtraction"""
    return tuple([x - y for x, y in zip(d1, d2)])


def add_tuple(d1, d2):
    """Add two tuples and return sum"""
    return tuple([x + y for x, y in zip(d1, d2)])


def apply_mask(data, mask):
    """
    Apply mask to data

    :param data: Input array
    :type data: MetrologyData/np.ndarray
    :param mask: Mask array
    :type mask: np.ndarray
    :return: Masked data, new start, new end
    :rtype: MetrologyData/np.ndarray, tuple, tuple
    """
    ret_data = data
    dims = get_dims(data)
    axes = dims.nonzero()[0]
    start_data = data.start_position_pix
    center_data = data.center_pix if isinstance(data, MetrologyData) else (0,) * mask.ndim
    st, en = get_default_positions(data, axes=axes)
    if np.any(mask):
        st, en = get_mask_positions(mask)
        slc = [slice(None)] * len(dims)
        if mask.ndim == 1:
            slc[-1] = slice(st[-1], en[-1])
        else:
            slc[axes[-2]] = slice(st[0], en[0])
            slc[axes[-1]] = slice(st[1], en[1])
        data_mask = data[tuple(slc)]
        ret_data = data_mask

    st = remove_tuple(st, center_data)
    st = add_tuple(st, start_data)
    en = remove_tuple(en, center_data)
    en = add_tuple(en, start_data)
    return ret_data, st, en


def get_default_positions(data, axes=[-2, -1]):
    """
    Get default mask positions from a dataset.

    :param data: Dataset
    :type data: np.ndarray
    :param axes: Axes along which to get start and end positions
    :type axes: list[int]
    :return: (start y, start x), (end y, end x)
    :rtype: tuple(float, float), tuple(float, float)
    """
    if data.ndim == 1:
        return (0,), (len(data),)
    elif data.ndim >= 2:
        return (0, 0), (np.size(data, axes[-2]), np.size(data, axes[-1]))


def get_mask_positions(mask):
    """
    Get mask positions from given 1D/2D mask.

    :param mask: Mask array
    :type mask: np.ndarray
    :return: (start y, start x), (end y, end x)
    :rtype: tuple(float, float), tuple(float, float)
    """
    if mask.ndim == 1:
        return (np.min(np.nonzero(mask)),), (np.max(np.nonzero(mask)),)
    else:
        max_mask = mask
        if mask.ndim > 2:
            for i in range(0, mask.ndim - 2):
                max_mask = np.any(max_mask,
                                  axis=0)  # maximum applicable mask elements by collapsing until last two dimensions
        v = np.transpose(np.nonzero(max_mask))
        mn = np.min(v, axis=0)
        mx = np.max(v, axis=0)
        return (mn[-2], mn[-1]), (mx[-2] + 1, mx[-1] + 1)


def get_instrument_details(h5Entry):
    """
    Get instrument details from a hdf5 entry (with NXentry as nexus class).

    :param h5Entry: H5 entry
    :type h5Entry: H5Group
    :return: Instrument details object selected from sql database
    :rtype: SelectResultsClass
    """
    retVal = None
    try:
        for key in h5Entry:
            if 'NX_class' in h5Entry[key].attrs and h5Entry[key].attrs['NX_class'] == 'NXinstrument':
                if 'instr_id' in h5Entry[key].attrs:
                    instr_id = h5Entry[key].attrs['instr_id']
                    if Instruments.selectBy(instrId=instr_id).count() > 0:
                        qinstr = Instruments.selectBy(instrId=instr_id)[0]
                        retVal = qinstr
    except:
        alertMsg('Error', 'Error getting instrument details')
    return retVal


def get_entries(h5Obj, first='Select entry'):
    """
    Get entries in the h5 object with nexus NX_class 'NXentry'.

    :param h5Obj: H5 object
    :type h5Obj: H5File
    :param first: First option in entries dropdown, e.g. 'Select entry'
    :type first: str
    :return: list of entries
    :rtype: list[str]
    """
    entries = [first]
    if h5Obj is not None:
        for idx in h5Obj:
            obj = h5Obj[idx]
            if 'NX_class' in obj.attrs and obj.attrs['NX_class'] == 'NXentry' and obj.name != '/StitchResults':
                entries.append(obj.name)
    return entries


def get_stitch_setups(h5Obj, entry, first='New setup'):
    """
    Get stitching setups from h5 object.

    :param h5Obj: H5 object
    :type h5Obj: H5Group / H5File
    :param entry: Entry name
    :type entry: str
    :param first: First option for setups dropdown, e.g. 'New setup'
    :type first: str
    :return: list of setups
    :rtype: list[str]
    """
    setups = [first]
    if h5Obj is not None and 'StitchResults' in h5Obj:
        h5sr = h5Obj['StitchResults']
        for it in h5sr:
            if 'NX_class' in h5sr[it].attrs and h5sr[it].attrs['NX_class'] == 'NXprocess' and h5sr[it].attrs['measurement_entry'] == entry:
                setups.append(h5sr[it].name)
    return setups


def load_class(class_name, locs):
    """
    Load class if not already loaded.

    :param class_name: Name of the class
    :type class_name: str
    :param locs: list of module locations
    :type locs: list[str]
    :return: Class object
    :rtype: Class
    """
    for loc in locs:
        dirname = os.path.dirname(getattr(importlib.import_module(loc), '__file__'))
        for importer, package_name, _ in pkgutil.iter_modules([dirname]):
            full_module_name = '%s.%s' % (loc, package_name)
            module = importlib.import_module(full_module_name)
            if hasattr(module, class_name):
                return getattr(module, class_name)
        return None


def parseDefValue(valStr):
    """
    Parse default strings. e.g.'N', 'NO', 'FALSE' are parsed to boolean False

    :param valStr: Input string
    :type valStr: str
    :return: Parsed object
    :rtype:
    """
    if type(valStr) != str:
        return valStr
    elif valStr.upper() in ['N', 'NO', 'FALSE']:
        return False
    elif valStr.upper() in ['Y', 'YES', 'TRUE']:
        return True
    elif valStr.upper() == 'NONE':
        return valStr
    else:
        try:
            import ast
            return ast.literal_eval(valStr)
        except:
            return valStr
    # return None


def closeH5(h5f):
    """
    Close hdf5 file object.

    :param h5f: File object
    :type h5f: h5py.File
    """
    try:
        h5f.close()
        h5f = None
    except:
        h5f = None


def alertMsg(title, msg):
    """
    Show alert dialog.

    :param title: Dialog title
    :type title: str
    :param msg: Alert message
    :type msg: str
    """
    import PyQt5.Qt as qt
    try:
        app = qt.QApplication.instance()
        if app is None:
            app = qt.QApplication([])
    except Exception as e:
        print('alertMsg <- commons')
        print(e)
    qt.QMessageBox.warning(None, title, msg)


def infoMsg(title, msg):
    """
    Show information message dialog.

    :param title: Dialog title
    :type title: str
    :param msg: Information message
    :type msg: str
    """
    import PyQt5.Qt as qt
    try:
        app = qt.QApplication.instance()
        if app is None:
            app = qt.QApplication([])
    except Exception as e:
        print('infoMsg <- commons')
        print(e)
    qt.QMessageBox.information(None, title, msg)


def questionMsg(parent=None, title="Title", msg="Yes/No?"):
    """
    Question dialog with Yes or No options.

    :param parent: Parent object
    :type parent: QWidget
    :param title: Dialog title
    :type title: str
    :param msg: Question or message
    :type msg: str
    :return: Clicked option
    :rtype: bool
    """
    answer = QMessageBox.question(parent,
                                  title,
                                  msg,
                                  QMessageBox.Yes | QMessageBox.No)
    if answer == QMessageBox.Yes:
        return True
    else:
        return False


def questionMsgAdv(parent=None, title="Title", msg="Yes/No/Cancel?"):
    """
    Advanced question dialog with Yes, No and Cancel options.

    :param parent: Parent object
    :type parent: QWidget
    :param title: Dialog title
    :type title: str
    :param msg: Question or message
    :type msg: str
    :return: Clicked option 0 = Cancel, 1 = No, 2 = Yes
    :rtype: int
    """
    answer = QMessageBox.question(parent,
                                  title,
                                  msg,
                                  QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
    if answer == QMessageBox.Yes:
        return 2
    elif answer == QMessageBox.No:
        return 1
    else:
        return 0


def save_file_dialog(start_dir, start_filter, file_formats,
                     title="Save as...", dialog=None):
    """
    Save file dialog with file formats.

    Function also returns the format and filter to cover the case where the
    same extension appears in multiple filters.

    :param start_dir: initial directory, optionally including the filename
    :type start_dir: str
    :param start_filter: initial filter
    :type start_filter: str
    :param file_formats: file formats
    :type file_formats: list[FileFormat]
    :param title: Title of the dialog
    :type title: str
    :param dialog: A function that creates a QT dialog
    :type dialog: Callable
    :return: (filename_list, file_format, filter), or `(None, None, None)` on cancel
    :rtype: tuple(list, FileFormat, str)
    """
    file_formats = sorted(set(file_formats), key=lambda w: (w.PRIORITY, w.DESCRIPTION))
    filters = [format_filter(f) for f in file_formats]

    if start_filter not in filters:
        start_filter = filters[0]

    if dialog is None:
        dialog = QFileDialog.getSaveFileName
    filename, filter = dialog(None, title, start_dir, ';;'.join(filters), start_filter)
    if not filename:
        return None, None, None

    if filter in filters:
        file_format = file_formats[filters.index(filter)]
    else:
        file_format = None
        filter = None

    return filename, file_format, filter


def open_multifile_dialog(start_dir, start_filter, file_formats,
                          add_all=True, title="Open...", dialog=None):
    """
    Open file dialog with file formats.

    Function also returns the format and filter to cover the case where the
    same extension appears in multiple filters.

    :param start_dir: initial directory, optionally including the filename
    :type start_dir: str
    :param start_filter: initial filter
    :type start_filter: str
    :param file_formats: file formats
    :type file_formats: list[FileFormat]
    :param title: Title of the dialog
    :type title: str
    :param dialog: A function that creates a QT dialog
    :type dialog: Callable
    :return: (filename_list, file_format, filter), or `(None, None, None)` on cancel
    :rtype: tuple(list, FileFormat, str)
    """
    file_formats = sorted(set(file_formats), key=lambda w: (w.PRIORITY, w.DESCRIPTION))
    filters = [format_filter(f) for f in file_formats]

    # add all readable files option
    if add_all:
        all_extensions = set()
        for f in file_formats:
            all_extensions.update(f.EXTENSIONS)
        file_formats.insert(0, None)
        filters.insert(0, "All readable files (*{})".format(
            ' *'.join(sorted(all_extensions))))

    if start_filter not in filters:
        start_filter = filters[0]

    if dialog is None:
        dialog = QFileDialog.getOpenFileNames
    filename, filter = dialog(
        None, title, start_dir, ';;'.join(filters), start_filter)
    if not filename:
        return None, None, None

    if filter in filters:
        file_format = file_formats[filters.index(filter)]
    else:
        file_format = None
        filter = None

    return filename, file_format, filter


def parseQInt(quser):
    """
    Parse string to integers, e.g. format '1-3,5' is parsed to [1,2,3,5]

    :param quser: Input string
    :type quser: str
    :return: Parsed integer array
    :rtype: list[int]
    """
    try:
        retArr = []
        if quser == '':
            return retArr
        else:
            A = quser.split(',')
            for a in A:
                if '-' in a:
                    an = a.split('-')
                    for i in range(int(an[0]), int(an[1]) + 1):
                        retArr.append(i)
                else:
                    retArr.append(int(a))
            return retArr
    except Exception as e:
        print('parseQInt <- util_functions')
        print(e)


def flip_data(data, axis, flip_motors=[]):
    """
    Flip data along given axis.

    :param data: Input dataset
    :type data: MetrologyData / np.ndarray
    :param axis: Axis
    :type axis: int
    :param flip_motors: Also flip selected motors, e.g. ['X', 'RY']
    :type flip_motors: list[str]
    :return: Flipped data
    :rtype: MetrologyData / np.ndarray
    """
    if isinstance(data, MetrologyData):
        index_list_prev = data.index_list
        data = np.flip(data, axis)
        index_list = data.index_list
        if data.init_shape is not None and data.init_shape[axis] > np.max(index_list[axis]):
            index_list[axis] = data.init_shape[axis] - np.array(index_list[axis]) - 1 # TODO: should be shifted by -1
        else:
            index_list[axis] = index_list_prev[axis]
        data._set_index_list(index_list)
        for axis_id in flip_motors:
            for m in data.motors:
                motor_id = 'motor_{}'.format(axis_id.upper())
                if m['name'] == motor_id and motor_id not in data.axis_values:
                    m['values'] = -1 * m['values']
    elif isinstance(data, np.ndarray):
        data = np.flip(data, axis)
    return data


def get_suplementary_output(options, data, module, keep_tag=False):
    ret_data = {}
    if not isinstance(options, (list, tuple, np.ndarray)):
        options = [options]

    if module == 'custom':
        for x in DEFAULT_DATA_NAMES:
            for option in options:
                key = x + '_' + option
                if key in data:
                    ret_data[key if keep_tag else x] = data[key]
    elif module == 'scan_data':
        scans = data['scan_data']
        ret_scans = {}
        for i, it in enumerate(scans):
            scan = scans[it]
            for x in DEFAULT_DATA_NAMES:
                for option in options:
                    key = x + '_' + option
                    if key in scan:
                        if it not in ret_scans:
                            ret_scans[it] = {}
                        ret_scans[it][key if keep_tag else x] = scan[key]
        if len(ret_scans) > 0:
            ret_data['scan_data'] = ret_scans
    return ret_data if len(ret_data) > 0 else None


def remove_suplementary_output(options, data, module):
    if not isinstance(options, (list, tuple, np.ndarray)):
        options = [options]
    if module == 'custom':
        for x in DEFAULT_DATA_NAMES:
            for option in options:
                key = x + '_' + option
                if key in data:
                    del data[key]
    elif module == 'scan_data':
        scans = data['scan_data']
        for i, it in enumerate(scans):
            scan = scans[it]
            for x in DEFAULT_DATA_NAMES:
                for option in options:
                    key = x + '_' + option
                    if key in scan:
                        del scan[key]
