# coding=utf-8
import numpy as np
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QScrollArea, QSizePolicy as Policy
from astropy import units as u
from astropy.units import Quantity
from silx.gui.data import DataViews

from pylost_widgets.config import config_params
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import DEFAULT_DATA_NAMES, MODULE_MULTI, MODULE_SINGLE, copy_items, \
    differentiate_heights, get_default_data_names, has_key_in_dict, integrate_slopes, stack_nparrays


# Important, if the __init__ in widgets package not loaded directly it will be called here


class PylostWidgets(OWWidget, openclass=True):
    """wrap __new__ method, title can be changed here"""
    # node = None
    scheme = None

    def __new__(cls, *args, captionTitle=None, **kwargs):
        klass = super().__new__(cls, *args, captionTitle=cls.name, **kwargs)
        manager = kwargs.get('signal_manager', None)
        if manager is not None:
            # cls.node = list(manager._SignalManager__node_outputs.keys())[0]
            cls.scheme = manager._SignalManager__workflow
            for node in cls.scheme._Scheme__nodes:
                if node.title.startswith(cls.name) and len(node.title) > len(cls.name):
                    node.title = cls.name
        return klass


class ModuleManager:

    def update_data_names(self, data_in, show_only_default=False, show_all_default_names=False, multiple=False):
        """
        Update data names in a orange pylost widget

        :param data_in: Input data
        :type data_in: dict
        :param show_only_default: Display / analyze only default data names and no custom data names, i.e. only slopes and heights
        :type show_only_default: bool
        :param show_all_default_names: Display / analyze all default data names, e.g. integrate slopes to get heights if not existing
        :type show_all_default_names: bool
        :param multiple: Flag about the data from multiple inputs
        :type multiple: bool
        """
        self.DATA_NAMES = []
        self.DATA_MODULES = []
        if data_in is not None and len(data_in) > 0:
            prev_module = data_in['module'] if 'module' in data_in else ''
            if show_all_default_names:
                for item in DEFAULT_DATA_NAMES:
                    self.DATA_NAMES.append(item)
            for item in DEFAULT_DATA_NAMES:
                if item not in self.DATA_NAMES and has_key_in_dict(item, data_in):
                    self.DATA_NAMES.append(item)
            if not show_only_default:
                if multiple:
                    for id in data_in:
                        self.add_custom_data_names(data_in[id])
                else:
                    self.add_custom_data_names(data_in)

            if has_key_in_dict('stitch_data', data_in) and has_key_in_dict('stitch_avg', data_in):
                self.DATA_MODULES.append('stitch_avg')
            if has_key_in_dict('stitch_data', data_in) and has_key_in_dict('stitched_scans', data_in):
                self.DATA_MODULES.append('stitch_data')
            if has_key_in_dict('scan_avg', data_in):
                self.DATA_MODULES.append('scan_avg')
            if has_key_in_dict('scan_data', data_in):
                self.DATA_MODULES.append('scan_data')

            if any(set(self.DATA_NAMES).intersection(data_in.keys())):
                self.DATA_MODULES.append('custom')
            elif multiple:
                for key in data_in:
                    if any(set(self.DATA_NAMES).intersection(data_in[key].keys())):
                        self.DATA_MODULES.append('custom')
                        break

            self.add_module_items(self.DATA_MODULES, prev_module=prev_module)

    def add_module_items(self, items, prev_module=''):
        """
        Add module items to the module dropdown, e.g. [custom, scan_data]

        :param items: All modules
        :type items: list
        :param prev_module: Module of last widget, if exists set to current widget
        :type prev_module: str
        """
        if hasattr(self, 'selModule') and len(items) > 0:
            self.selModule.clear()
            self.selModule.addItems(items)
            if prev_module != '' and prev_module in items:
                self.module = prev_module
            else:
                self.module = self.selModule.currentText()

    def add_custom_data_names(self, data_in):
        """
        Add custom data names (e.g. not slopes_x, slopes_, height) to all the data names. The datasets with names added here will be processed.

        :param data_in: Input dictionary data
        :type data_in: dict
        """
        if 'CUSTOM_DATA_NAMES' in data_in and any(data_in['CUSTOM_DATA_NAMES']):
            for item in data_in['CUSTOM_DATA_NAMES']:
                if has_key_in_dict(item, data_in) and item not in self.DATA_NAMES:
                    self.DATA_NAMES.append(item)

    @staticmethod
    def update_module(cur, new):
        """
        Update module while loading multiple inputs. If one of the module is in MODULE_MULTI (i.e. has repeated scans), the final module witll be multi.

        :param cur: Current module
        :type cur: str
        :param new: New module from input
        :type new: str
        :return: Updated module
        :rtype: str
        """
        if cur == '':
            return new
        elif cur in MODULE_MULTI:
            return cur
        elif new in MODULE_MULTI and cur in MODULE_SINGLE:
            return new
        else:
            return cur

    # TODO: Let us start using multi input for visualization only
    def update_input_modules(self, data, multiple=False):
        """
        Update input modules for single or multiple input channels.

        :param data: Input data
        :type data: dict
        :param multiple: Flag about data from multiple input channels
        :type multiple: bool
        """
        if multiple:
            for key in data:
                self.update_input_modules_single(data[key])
        else:
            self.update_input_modules_single(data)

    @staticmethod
    def update_input_modules_single(data):
        """
        Update module if not already present.

        :param data: Input data
        :type data: dict
        """
        if 'module' in data:
            return
        for key in MODULE_MULTI + MODULE_SINGLE:
            if key in data:
                data['module'] = key
                return
        if any(set(DEFAULT_DATA_NAMES).intersection(data.keys())):
            data['module'] = 'custom'

    def get_data_by_module(self, data, module, multiple=False):
        """
        Get data by given module for single or multiple input channels.

        :param data: Input data
        :type data: dict
        :param module: Given module
        :type module: str
        :param multiple: Flag about data from multiple input channels
        :type multiple: bool
        :return: Formatted data object
        :rtype: dict
        """
        retObj = {}
        if multiple:
            for id in data:
                ret, module_ret = self.get_data_by_module_default(data[id], module)
                retObj[(id, module_ret)] = ret
        else:
            retObj, module_ret = self.get_data_by_module_default(data, module)
        return retObj

    def get_data_by_module_default(self, data, module):
        """
        Get data by module. Currently 'custom', 'scan_data' modules are only used.
        Module 'custom' is used for a single scan data loaded through Data (File) single/sequence.
        Module 'scan_data' is used for multiple scan data loaded through Data (H5) or Data (scans).
        Stitching will save the result in same module type. Widgets such as AverageScans takes 'scan_data' input and outputs 'custom' module

        :param data: Input data
        :type data: dict
        :param module: Given module
        :type module: str
        :return:
        """
        ret = {}
        if module is None:
            if 'module' in data:
                module = data['module']
            else:
                module = 'custom'
            if getattr(self, 'module', None) is not None:
                self.module = self.update_module(self.module, module)
        if module == 'stitch_avg' and 'stitch_data' in data and 'stitch_avg' in data['stitch_data']:
            ret = data['stitch_data']['stitch_avg']
        if module == 'stitch_data' and 'stitch_data' in data and 'stitched_scans' in data['stitch_data']:
            ret = data['stitch_data']['stitched_scans']
        if module == 'scan_avg' and 'scan_avg' in data:
            ret = data['scan_avg']
        if module == 'scan_data' and 'scan_data' in data:
            ret = data['scan_data']
        if module == 'custom':
            ret = data
        return ret, module

    def set_data_by_module(self, data, module, val, multiple=False):
        """
        Set data by given module for single or multiple input channels

        :param data: Output data
        :type data: dict
        :param module: Given module
        :type module: str
        :param val: Values to set
        :type val: dict
        :param multiple: Flag about data from multiple input channels
        :type multiple: bool
        """
        if multiple:
            for key in val:
                if key in data:
                    module_key = data[key]['module'] if 'module' in data[key] else module
                    self.set_data_by_module_single(data[key], module_key, val[key])
        else:
            self.set_data_by_module_single(data, module, val)

    @staticmethod
    def set_data_by_module_single(data, module, val):
        """
        Save data by module. Currently 'custom', 'scan_data' modules are only used.
        :param self:
        :param data: Output data
        :param module: Save module
        :param val: Values to save
        :return:
        """
        if module == 'stitch_avg':
            if 'stitch_data' not in data:
                data['stitch_data'] = {}
            data['stitch_data']['stitch_avg'] = val
        if module == 'stitch_data':
            if 'stitch_data' not in data:
                data['stitch_data'] = {}
            data['stitch_data']['stitched_scans'] = val
        if module == 'scan_avg':
            data['scan_avg'] = val
        if module == 'scan_data':
            data['scan_data'] = val
        if module == 'custom':
            copy_items(val, data)
        data['module'] = module


class ScanDataManager:
    """Manages the actions on scan data such integration/differentiations on slope/height data respectively."""

    def __init__(self, scan={}):
        """
        Initialization for class

        :param scan: Selected scan
        :type scan: dict
        """
        super(ScanDataManager).__init__()
        self.scan = scan

    def full_dataset(self, scan, multiple=False, isShapeRemoved=False):
        """
        Fill missing datasets, if slopes are present integrate to get heights, else if heights are present differentiate to get slopes

        :param scan: Scan datasets
        :type scan: dict
        :param multiple: Flag about data from multiple input channels
        :type multiple: bool
        :param isShapeRemoved: Flag about shape (e.g. sphere, ellipse) already removed from the scan data
        :type isShapeRemoved: bool
        :return: Full data with both slopes and heights
        :rtype: dict
        """
        if multiple:
            scan_vis = {}
            for key in scan:
                id = key[0]
                scan_vis[id] = self.full_dataset_single(scan[key], isShapeRemoved=isShapeRemoved)
            return scan_vis
        else:
            return self.full_dataset_single(scan, isShapeRemoved=isShapeRemoved)

    def full_dataset_single(self, scan, isShapeRemoved=False):
        """
        Fill missing datasets, if slopes are present integrate to get heights, else if heights are present differentiate to get slopes

        :param scan: Scan datasets
        :type scan: dict
        :param isShapeRemoved: Flag about shape (e.g. sphere, ellipse) already removed from the scan data
        :type isShapeRemoved: bool
        :return: Full data with both slopes and heights
        :rtype: dict
        """
        scan_vis = {}
        x = y = None
        copy_items(scan, scan_vis, deepcopy=True)
        # scan_vis = {item: scan[item] for item in scan}
        try:
            pix_size = self.pix_size if hasattr(self, 'pix_size') else None
            for key in ['slopes_x', 'height']:
                if key in scan and isinstance(scan[key], MetrologyData):
                    if any(scan[key].dim_detector):
                        pix_size = scan[key].pix_size_detector
                    else:
                        axis_vals = scan[key].get_axis_val_items()
                        x = axis_vals[-1] if len(axis_vals) > 0 else None
                        y = axis_vals[-2] if len(axis_vals) > 1 else None
            if 'slopes_x' in scan:
                if 'slopes_y' in scan:
                    if 'height' not in scan:
                        scan_vis['height'] = integrate_slopes(scan['slopes_x'], scan['slopes_y'], x=x, y=y,
                                                              pix_sz=pix_size, interpolate_nans=self.interpolate,
                                                              method='frankot_chellappa', isShapeRemoved=isShapeRemoved)
                        scan_vis['height'] = self.heights_to_metrologydata(scan, scan_vis, pix_size, x=x, y=y)
                else:
                    if 'height' not in scan:
                        scan_vis['height'] = integrate_slopes(scan['slopes_x'], x=x, y=y, pix_sz=pix_size,
                                                              method='trapz')
                        scan_vis['height'] = self.heights_to_metrologydata(scan, scan_vis, pix_size, x=x, y=y)
            if 'height' in scan:
                if 'slopes_x' not in scan:
                    scan_vis['slopes_x'], scan_vis['slopes_y'] = differentiate_heights(scan['height'], x=x, y=y,
                                                                                       pix_sz=pix_size, method='grad')
                    scan_vis['slopes_x'], scan_vis['slopes_y'] = self.slopes_to_metrologydata(scan, scan_vis, pix_size,
                                                                                              x=x,
                                                                                              y=y)
        except Exception as e:
            self.Error.unknown(repr(e))
        return scan_vis

    @staticmethod
    def heights_to_metrologydata(scan, scan_vis, pix_size, x=None, y=None):
        """
        If slopes are in MetrologyData format, apply its attributes to integrated heights numpy array and convert to MetrologyData format

        :param scan: Input scan datasets
        :type scan: dict
        :param scan_vis: Output scan datasets with heights data
        :type scan_vis: dict
        :param pix_size: Pixel size
        :type pix_size: list
        :param x: Axis x positions
        :type x: Quantity
        :param y: Axis y positions
        :type y: Quantity
        :return: Heights data
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

    @staticmethod
    def slopes_to_metrologydata(scan, scan_vis, pix_size, x=None, y=None):
        """
        If heights are in MetrologyData format, apply its attributes to differentiated slopes numpy arrays and convert to MetrologyData format

        :param scan: Input scan datasets
        :type scan: dict
        :param scan_vis: Output scan datasets with slopes data
        :type scan_vis: dict
        :param pix_size: Pixel size
        :type pix_size: list
        :param x: Axis x positions
        :type x: Quantity
        :param y: Axis y positions
        :type y: Quantity
        :return: Slopes x, Slopes y
        :rtype: (MetrologyData, MetrologyData)
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


class DataVisualizationManager:
    """Manages data visualization actions"""

    def update_tabs(self, data_in, show_all_default_names=False, show_only_default=False, multiple=False):
        """
        Update tabs (remove and add tabs with given data names), in the display widgets like OWVisualize

        :param data_in: Input data
        :type data_in: dict
        :param show_all_default_names: Show all default data names, i.e. both slopes and heights
        :type show_all_default_names: bool
        :param show_only_default: Show only default data but no custom data
        :type show_only_default: bool
        :param multiple: Flag about data from multiple inputs
        :type multiple: bool
        """
        self.update_data_names(data_in, show_only_default=show_only_default,
                               show_all_default_names=show_all_default_names, multiple=multiple)
        self.dataViewers = {}
        for i in np.arange(self.tabs.count())[::-1]:
            self.tabs.removeTab(i)
        for item in self.DATA_NAMES:
            if item in self.DEFAULT_DATA_NAMES and show_all_default_names:
                self.add_new_tab(item)
            elif data_in is not None and has_key_in_dict(item, data_in):
                self.add_new_tab(item)

    def add_new_tab(self, name):
        """
        Add a new tab in the TabWidget with the given name. The tab by default has only a silx DataViewer

        :param name: Name of the tab
        :type name: str
        """
        from pylost_widgets.util.DataViewerFrameOrange import DataViewerFrameOrange
        dv = DataViewerFrameOrange(self)
        cmap = 'Greys_r' if 'slopes' in name else 'turbo'
        dv.setCmapName(cmap)
        self.dataViewers[name] = dv
        self.tabs.addTab(dv, name)

    def load_viewer(self, scan, show_mask=False, multiple=False):
        """
        Load data viewers with the given slopes/height/custom data from given scan

        :param scan: Given scan dictionary data, can have multiple entries like slopes_x, height
        :type scan: dict
        :param show_mask: Show mask tools in the data viewer toolbar
        :type show_mask: bool
        :param multiple: Flag about data from multiple inputs
        :type multiple: bool
        """
        try:
            for i, item in enumerate(self.DATA_NAMES):
                if item not in self.dataViewers:
                    return
                if multiple:
                    enabled = True
                    temp = None
                    scan_mult = {}
                    for id in scan:
                        if len(scan) == 1 and item in scan[id]:
                            scan_mult[item] = scan[id][item]
                            break
                        if item in scan[id]:
                            temp = scan[id][item] if isinstance(scan[id][item], MetrologyData) else None
                            stack_nparrays(scan[id], scan_mult, scan[id], scan_mult, item,
                                           start_pos_keys='zeros')  # TODO: should loading start position keys allowed here?
                        elif has_key_in_dict(item, scan):  # This item is available in another id. Add nan array.
                            temp = scan_mult[item] if item in scan_mult and isinstance(scan_mult[item],
                                                                                       MetrologyData) else None
                            stack_nparrays({item: np.array([np.nan])}, scan_mult, None, scan_mult, item,
                                           start_pos_keys='zeros')
                        if temp is not None:
                            scan_mult[item] = temp.copy_to(scan_mult[item])
                    if item in scan_mult:
                        self.dataViewers[item].setData(scan_mult[item])
                        self.tabs.setTabEnabled(i, True)
                elif item in scan:
                    enabled = True
                    self.dataViewers[item].setData(scan[item])
                    self.tabs.setTabEnabled(i, True)
                else:
                    enabled = False
                    self.tabs.setTabEnabled(i, False)

                if enabled:
                    try:
                        views = self.dataViewers[item].currentAvailableViews()
                        view_types = [type(x) for x in views]
                        if DataViews._ImageView in view_types:
                            self.dataViewers[item].setDisplayMode(DataViews.IMAGE_MODE)
                        elif DataViews._Plot1dView in view_types:
                            self.dataViewers[item].setDisplayMode(DataViews.PLOT1D_MODE)
                        elif DataViews._RawView in view_types:
                            self.dataViewers[item].setDisplayMode(DataViews.RAW_MODE)

                        if show_mask:
                            self.dataViewers[item].showMaskTools()
                    except:
                        pass
        except Exception as e:
            print(e)

    def clear_viewers(self):
        """
        Clear the data viewers across all tabs
        """
        for i, item in enumerate(self.DATA_NAMES):
            self.tabs.setTabEnabled(i, False)
            if item in self.dataViewers:
                self.dataViewers[item].setData(None)


class InitWidgetManager:
    """Widget initialization actions"""

    def init_info(self, module=True, module_callback=None, scans=False, scans_callback=None):
        """
        Initialize information box (loacted at the top of most widgets), module dropdown, scans dropdown (for module 'scan_data').

        :param module: Module name
        :type module: bool
        :param module_callback: Module callback function if any
        :type module_callback: function
        :param scans: Init scans?
        :type scans: bool
        :param scans_callback: Callback function for scans dropdown, if exists
        :type scans_callback: func
        :return: Orange widget horizontal box with given UI elements
        :rtype: gui.hBox
        """
        box = gui.hBox(self.controlArea, "Info", stretch=1)
        self.info_scroll(box)
        if module:
            self.init_module(box, callback=module_callback)
        if scans:
            self.init_scans(box, callback=scans_callback)
        return box

    def info_scroll(self, box):
        """
        Information scrollbar at the top of all widgets containing impormation about the applied processes on the data upstream.

        :param box: Parent box to attach this widget
        :type box: QWidget
        """
        box1 = gui.hBox(box, "", stretch=5)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.infoInput = gui.widgetLabel(None, "No data on input yet, waiting to get something.")
        scroll.setWidget(self.infoInput)
        scroll.setFrameShape(QFrame.NoFrame)
        box1.layout().addWidget(scroll)

    def init_module(self, box, callback=None):
        """
        Initialize module selection dropdown, e.g. 'scan_data' for data with many repeated scans, 'custom' for any random loaded data.

        :param box: Parent box to attach this widget
        :type box: QWidget
        """
        self.selModule = gui.comboBox(box, self, "module", label='Module:', callback=callback,
                                      # callback=self.change_module,
                                      sendSelectedValue=True, orientation=Qt.Horizontal, stretch=1,
                                      sizePolicy=(Policy.Fixed, Policy.Fixed))

    def init_scans(self, box, callback=None):
        """
        Initialize scan selection dropdown. Useful only if many repeated scans are present in the data.

        :param box: Parent box to attach this widget
        :type box: QWidget
        """
        self.selScan = gui.comboBox(box, self, 'scan_name', label='Scans:', callback=callback,
                                    # callback=self.change_scan,
                                    orientation=Qt.Horizontal, sendSelectedValue=True, stretch=1,
                                    sizePolicy=(Policy.Fixed, Policy.Fixed))


class PylostBase(InitWidgetManager, ModuleManager, ScanDataManager, DataVisualizationManager):
    """Base class for all orange pylost widgets"""

    def __init__(self):
        """Initialization"""
        InitWidgetManager.__init__(self)
        ModuleManager.__init__(self)
        ScanDataManager.__init__(self)
        DataVisualizationManager.__init__(self)

        self.data_in = {}
        self.data_out = {}
        self.pix_size = [1, 1]
        self.DEFAULT_DATA_NAMES = get_default_data_names()
        self.DATA_NAMES = []
        self.DATA_MODULES = []

    def set_data(self, data, id=None, update_tabs=False, update_names=False, deepcopy=True, **kwargs):
        """
        Update parameters and visualization (if applies) based on the input data

        :param data: Input data
        :type data: dict
        :param id: Id of the input, if the channel is multi-input
        :type id: str
        :param update_tabs: Flag to update data visualization tabs
        :type update_tabs: bool
        :param update_names: Flag to update dataset names (e.g. if 'height' key is present in data object, add it to the dataset names)
        :type update_names: bool
        :param deepcopy: Deepcopy dictionary items
        :type deepcopy: bool
        :param kwargs: Additional arguments
        :type kwargs: any
        """
        self.clear_messages()
        self.data_in = {}
        self.data_out = {}
        if data is not None:
            self.data_in = data
            copy_items(data, self.data_out, deepcopy=deepcopy)
            if update_names:
                super().update_data_names(data, **kwargs)
            if update_tabs:
                super().update_tabs(data, **kwargs)
            self.load_data()
        else:
            self.infoInput.setText("No data on input, waiting to get something.")
            try:
                self.selModule.clear()
            except AttributeError:
                # print('no module set')
                pass
            if update_names:
                super().update_data_names(None)
            if update_tabs:
                super().update_tabs(None)
                self.clear_viewers()

    def load_data(self, multi=False):
        """
        Initialize information box at the top with comments from previous widgets and add file name to the window title if available

        :param multi: Is the data from multi input channel?
        :type multi: bool
        """
        try:
            if multi:
                cmt = ''
                fnames = ''
                for key in self.data_in:
                    if 'comment_log' in self.data_in[key]:
                        # cmt += '{} :\n {}\n'.format(key, self.data_in[key]['comment_log'])
                        cmt += '{} link {} :\n\t{}\n'.format(self.name, key,
                                                             self.data_in[key]['comment_log'].replace('\n', '\n\t'))
                    if 'filename' in self.data_in[key]:
                        fnames += '{};'.format(self.data_in[key]['filename'])
                self.infoInput.setText(cmt)
                self.data_in['comment_log'] = cmt
                if fnames != '':
                    self.setWindowTitle('{} ({})'.format(self.name, fnames))
            else:
                if 'comment_log' in self.data_in:
                    self.infoInput.setText(self.data_in['comment_log'])
                if 'filename' in self.data_in:
                    self.setWindowTitle('{} ({})'.format(self.name, self.data_in['filename']))
                self.pix_size = self.data_in['pix_size'] if 'pix_size' in self.data_in else [1, 1]
        except Exception as e:
            self.Error.unknown(repr(e))

    def update_comment(self, comment, prefix=''):
        """
        Update comment text, shown as info_message and also forwarded to next widget as log text

        :param comment: Comment text
        :type comment: str
        :param prefix: Prefix text for comment
        :type prefix: str
        """
        if prefix == '':
            prefix = self.name
        cmt = self.data_in['comment_log'] + '\n' if 'comment_log' in self.data_in else ''
        self.data_out['comment_log'] = cmt + '\n{}: {}'.format(prefix, comment)

    def change_module(self):
        """Callback after changing the module (if many modules are present)"""
        self.selScan.clear()
        self.clear_viewers()
        module_data = self.get_data_by_module(self.data_in, self.module)
        if self.module in MODULE_MULTI:
            self.selScan.parent().show()
            if len(module_data) > 0:
                self.selScan.setEnabled(True)
                self.selScan.addItems(list(module_data.keys()))
        else:
            self.selScan.parent().hide()
        self.apply()

    def apply(self):
        self.apply_scans()

    def change_scan(self):
        """Callback after changing the scan (if scans dropdown is available"""
        self.update_viewer()

    def update_viewer(self):
        """Update data viewer (if exists) with loaded data. Data viewer is also updated if module or scan is changed"""
        module_data = self.get_data_by_module(self.data_out, self.module)
        if self.module in MODULE_MULTI:
            curScan = self.selScan.currentText()
            self.load_viewer(module_data[curScan], show_mask=False)
        elif self.module in MODULE_SINGLE:
            self.load_viewer(module_data, show_mask=False)

    @staticmethod
    def get_detector_dimensions(Z):
        """
        Get detector dimentions of a dataset which is either MetrologyData or numpy.ndarray. For numpy data typically last one or two dimensions are used as detector dimensions.

        :param Z: Input dataset
        :type Z: MetrologyData or numpy.ndarray
        :return: Detector dimensions
        :rtype: numpy.array
        """
        dims = [False] * Z.ndim
        dims[-1] = True
        if Z.ndim > 1:
            dims[-2] = True
        if isinstance(Z, MetrologyData) and np.any(Z.dim_detector):
            dims = Z.dim_detector
        dims = np.array(dims)
        return dims

    def apply_scans(self, autoclose=config_params.DEFAULT_CLOSE_WIDGETS_AFTER_APPLY):
        """Callback for apply button in widgets where applicable. Can be reimplmented in the widgets.
        Applies the module if necessary (e.g. if module is scan_data, it will loop over scans and calls apply_scan for each scan)"""
        try:
            comment = ''
            self.clear_messages()
            self.data_out = {}
            copy_items(self.data_in, self.data_out)
            module_data = super().get_data_by_module(self.data_in, self.module)
            if self.module in MODULE_MULTI:
                scans_fit = {}
                for it in module_data:
                    scan = module_data[it]
                    scan_fit, comment = self.apply_scan(scan, scan_name=it, comment=comment)
                    scans_fit[it] = scan_fit
                super().set_data_by_module(self.data_out, self.module, scans_fit)
            elif self.module in MODULE_SINGLE:
                scan_fit, comment = self.apply_scan(module_data)
                super().set_data_by_module(self.data_out, self.module, scan_fit)
            self.update_comment(comment)
            self.setStatusMessage(comment)
            self.info.set_output_summary(comment)
            if autoclose:
                self.close()
        except Exception as e:
            print(e)
            self.Error.unknown(repr(e))

    def apply_scan(self, scan, scan_name=None, comment=''):
        """
        Apply widget functionality for a single scan.

        :param scan: Scan dictionary with height/slope information
        :type scan: dict
        :param scan_name: Name of the scan, useful if many repeated scans are present
        :type scan_name: str
        :param comment: Comment text
        :type comment: str
        :return: (Result dictionary with processed scan data, comment)
        :rtype: (dict, str)
        """
        scan_result = {}
        copy_items(scan, scan_result)
        for i, item in enumerate(self.DATA_NAMES):
            if item in scan:
                scan_result[item], comment = self.apply_scan_item(scan[item], comment=comment, item=item)

        return scan_result, comment

    def apply_scan_item(self, Z, comment='', item=None):
        """
        Apply the operation for a scan item, e.g. height. This loops over non detector dimensions, to apply an operation (e.g. rotation, filter) just on the detector dimensions.
        Can be reimplemented in subclasses differently for some operations.

        :param Z: Scan item like slopes_x or height
        :type Z: Numpy ndarray or MetrologyData
        :param comment: Comment text
        :type comment: str
        :param item: Type of dataset 'height' / 'slopes_x' / 'slopes_y' or another custom dataset name
        :type item: str
        :return: Processed dataset, comment text
        :rtype: (MetrologyData/numpy.ndarray, str)
        """
        Zout = np.full_like(Z, fill_value=np.nan)
        dims = self.get_detector_dimensions(Z)
        idx_full = np.asarray([slice(None)] * Z.ndim)
        shp_non_detctor = tuple(np.asarray(Z.shape)[np.invert(dims)]) if any(dims) and False in dims else ()
        for idx in np.ndindex(shp_non_detctor):
            if any(shp_non_detctor):
                idx_full[np.invert(dims)] = np.asarray(idx)
            Zout[tuple(idx_full)], comment = self.apply_scan_item_detector(Z[tuple(idx_full)], comment)
        return Zout, comment

    def apply_scan_item_detector(self, Zdet, comment=''):
        """
        Apply the widget operation over only the detector dimensions of the dataset (e.g. height). All other dimensions (e.g. subaperture dimension) should be looped in the parent function

        :param Zdet: Dataset like slopes_x or height, sliced along detector dimensions
        :type Zdet: MetrologyData / numpy.ndarray
        :param comment: Comment text
        :type comment: str
        :return: Processed dataset, comment
        :rtype: (MetrologyData/numpy.ndarray, str)
        """
        # To implement in sub classes
        return Zdet, comment
