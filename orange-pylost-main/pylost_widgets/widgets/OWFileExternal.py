"""
OWFileExternal is copied from orange native OWFile and adapted to pylost use
"""

import ast
import logging
import os
from warnings import catch_warnings
import numpy as np
import re

from pathlib import Path

from Orange.data import Table
from Orange.data.io import class_from_qualified_name, FileFormat
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Output
from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSlot
from PyQt5.QtCore import Qt, QSize, QPoint
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QGridLayout, QDialog
from PyQt5.QtWidgets import \
    QStyle, QSizePolicy as Policy, QAbstractItemView, QLabel, QTabWidget, QLineEdit
from astropy.units import Quantity
from functools import partial
from orangewidget.settings import Setting
from orangewidget.utils.filedialogs import RecentPathsWComboMixin, RecentPath, open_filename_dialog
from orangewidget.utils.signals import Input
from orangewidget.widget import Msg
from typing import List

from pylost_widgets.config import config_params
from pylost_widgets.util import MetrologyData
from pylost_widgets.util.DataViewerFrameOrange import DataViewerFrameOrange
from pylost_widgets.util.DictionaryTree import DictionaryTreeWidget
from pylost_widgets.util.FileSeqLoader import FileSeqLoader
from pylost_widgets.util.Task import Task
from pylost_widgets.util.ow_filereaders import *
from pylost_widgets.scripts.file_formats import *
from pylost_widgets.util.resource_path import resource_path
from pylost_widgets.util.util_functions import get_import_data_names, copy_items, walk_dict, fill_dict, \
    questionMsgAdv, get_default_data_names, get_dict_item, open_multifile_dialog, walk_dict_by_type, questionMsg

log = logging.getLogger(__name__)

qtCreatorFile = resource_path(os.path.join("gui", "dialog_import.ui"))  # Enter file here.
UI_import, QtBaseClass = uic.loadUiType(qtCreatorFile)
qtaddnl = resource_path(os.path.join("gui", "dialog_import_additional_data.ui"))  # Enter file here.
UI_import_addnl, QtBaseClass = uic.loadUiType(qtaddnl)
qtmd = resource_path(os.path.join("gui", "dialog_metrology_data.ui"))  # Enter file here.
UI_met_data, QtBaseClass = uic.loadUiType(qtmd)

import concurrent.futures
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke

# Import user file format script paths
from pylost_widgets.util.util_scripts import import_paths

import_paths(param_name='FILE_FORMAT_PATH')

from pylost_widgets.widgets._PylostBase import PylostWidgets


class OWFileExternal(PylostWidgets, RecentPathsWComboMixin):
    """File loader widget"""
    name = 'Data (File)'
    description = 'Loads data from external file'
    icon = "../icons/import.svg"
    priority = 11

    class Inputs:
        data = Input('data', object, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = True

    SEARCH_PATHS = []  # TODO: set default paths

    recent_paths: List[RecentPath]

    recent_paths = Setting([])
    enable_multiselect = Setting(False)
    filename_list = Setting([])

    class Information(widget.OWWidget.Information):
        info = Msg("Info:\n{}")

    class Warning(widget.OWWidget.Warning):
        file_too_big = Msg("The file is too large to load automatically. Press Reload to load.")
        load_warning = Msg("Read warning:\n{}")

    class Error(widget.OWWidget.Error):
        incorrect_input = Msg(
            "Invalid input data. Data type of input can only be in (dict, tuple, np.ndarray, Table(Orange)).")
        file_not_found = Msg("File not found.")
        missing_reader = Msg("Missing reader.")
        unknown = Msg("Read error:\n{}")

    class NoFileSelected:
        pass

    def __init__(self):
        """
        Initialize GUI for file tree viewer and data visualizer. Two file viewers are used one for loaded data (input),
        and second (output) for data formatted to orange-pylost compatible mode (e.g. using MetrologyData)
        """
        super().__init__()
        self._clean_recentpaths()
        RecentPathsWComboMixin.__init__(self)
        self.data_out = {}
        self.data_in = {}
        self.source = 0
        self.filename_seq = []
        self.DEFAULT_DATA_NAMES = get_default_data_names()
        self.default_axis_names = ['Motor', 'Y', 'X']
        self.default_dim_detector = [-2, -1]
        if self.last_path() is not None and any(self.last_path()):
            self.add_path(self.last_path())

        self.controlArea.setMinimumWidth(400)

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, "Load", margin=10, orientation=layout, addSpace=True, stretch=1)
        lbl = gui.widgetLabel(None, 'Input file: ')
        layout.addWidget(lbl, 0, 0)
        self.file_combo.setMinimumWidth(150)
        self.file_combo.setSizePolicy(Policy.Fixed, Policy.Fixed)
        self.file_combo.activated[int].connect(self.select_file)
        layout.addWidget(self.file_combo, 0, 1)

        self.file_button = gui.button(None, self, '...', callback=self.browse_file_seq, autoDefault=False)
        self.file_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(self.file_button, 0, 2)

        self.reload_button = gui.button(None, self, "Reload", callback=self.load_seq, autoDefault=False)
        self.reload_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(self.reload_button, 0, 3)

        self.stitching_folder = gui.button(None, self, "Auto seq", callback=self.load_stitching_root, autoDefault=False)
        self.stitching_folder.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.stitching_folder.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(self.stitching_folder, 1, 2)

        self.btnAbort = gui.button(None, self, 'Abort', callback=self.abort, stretch=1, autoDefault=False)
        self.btnAbort.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(self.btnAbort, 1, 3)
        self.btnAbort.hide()

        box = gui.vBox(self.controlArea, "Info", stretch=1)
        self.infolabel = gui.widgetLabel(box, 'No data loaded.')
        self.warnings = gui.widgetLabel(box, '')

        # Dictionary view
        box = gui.vBox(self.controlArea, "File viewer", stretch=8)
        self.tabs = QTabWidget(self)
        # Tab input
        tabBox = gui.vBox(box)
        box1 = gui.hBox(tabBox)
        gui.widgetLabel(box1, 'Send to output: ')
        outall_button = gui.button(box1, self, 'Import all', callback=self.sendOutputAll, autoDefault=False)
        outall_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        outsel_button = gui.button(box1, self, 'Import selected', callback=self.sendOutput, autoDefault=False)
        outsel_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        gui.checkBox(tabBox, self, 'enable_multiselect', 'Enable multi selection', callback=self.check_multiselect)
        self.__treeViewerIn = DictionaryTreeWidget(self, None)
        self.__treeViewerIn.itemClicked.connect(self.displayDataIn)
        tabBox.layout().addWidget(self.__treeViewerIn)
        self.tabs.addTab(tabBox, 'Input')

        # Tab output
        tabBox = gui.vBox(box)
        box1 = gui.hBox(tabBox)
        gui.widgetLabel(box1, 'Output: ')
        # TODO: Need for additional data??
        # addnl = gui.button(box1, self, 'Additional data', callback=self.additionalData)
        # addnl.setSizePolicy(Policy.Maximum, Policy.Fixed)
        clear_output = gui.button(box1, self, 'Clear output', callback=self.clearOutput)
        clear_output.setSizePolicy(Policy.Maximum, Policy.Fixed)
        box2 = gui.hBox(tabBox)
        self.infoLoad = gui.widgetLabel(box2,
                                        "Select <a href='xyz'>custom datasets</a> for data analysis (only of type MetrologyData).",
                                        stretch=3)
        self.infoLoad.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        self.infoLoad.linkActivated.connect(self.select_custom_data)
        self.done_custom_select = gui.button(box2, self, 'Done', callback=self.doneSelection)
        self.done_custom_select.setSizePolicy(Policy.Maximum, Policy.Fixed)
        self.done_custom_select.hide()
        box3 = gui.hBox(tabBox)
        self.infoCustomData = gui.widgetLabel(box3, "", stretch=3)
        self.infoCustomData.setWordWrap(True)
        self.__treeViewerOut = DictionaryTreeWidget(self, None)
        self.__treeViewerOut.itemClicked.connect(self.displayDataOut)
        self.__treeViewerOut.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.__treeViewerOut.customContextMenuRequested.connect(self.rightClickTreeItem)
        tabBox.layout().addWidget(self.__treeViewerOut)
        self.tabs.addTab(tabBox, 'Output')
        box.layout().addWidget(self.tabs)

        # Data viewer
        box = gui.vBox(self.mainArea, "Data viewer")
        self.__dataViewer = DataViewerFrameOrange(self, editInfo=True, show_mask=False)
        box.layout().addWidget(self.__dataViewer)

        # Convert to metroogy data
        self.menuMetData = QtWidgets.QMenu()
        actionMetData = self.menuMetData.addAction("Convert to Metrology Data")
        actionMetData.triggered.connect(self.convToMetrologyData)

        self.Outputs.data.send(None)

        self._task = None
        self._executor = ThreadExecutor()
        self.__dataViewer.infoChanged.connect(self.send_new_info)

        self.return_pressed = False

        self.pixel_size = None

        self.updatemotors = None

    def _clean_recentpaths(self):
        pathlist = []
        for i, item in enumerate(self.recent_paths):
            if i > 20:
                break
            if Path(item.abspath).exists():
                pathlist.append(item)
        self.recent_paths = pathlist

    @staticmethod
    def _format_vis(val, fmt='{}'):
        """
        Format visualization text

        :param val: Values of various python data types
        :type val: tuple/list/np.ndarray
        :param fmt: Visualization format (e.g. show 3 floating points after decimal)
        :return: Formatted string of val
        :rtype: str
        """
        if val is None:
            return ''
        elif isinstance(val, (tuple, list, np.ndarray)):
            return ' x '.join(fmt.format(x) for x in val)
        else:
            return fmt.format(val)

    def send_new_info(self):
        if self.pixel_size is not None:
            for item in self.data_out.values():
                if isinstance(item, MetrologyData):
                    pixel_size = self._format_vis(item.pix_size_detector, fmt='{:.6f}')
                    break
            if pixel_size != self.pixel_size:
                cmt = str(self.data_out['comment_log']) + '\n' if 'comment_log' in self.data_out else ''
                cmt = cmt + f'\nNew pixel size set: {self.pixel_size} --> {pixel_size}'
                self.data_out['comment_log'] = cmt
        self.Outputs.data.send(self.data_out)

    def abort(self):
        """Callback for abort button. File loading is done in another thread and when aborted the thread is cancelled"""
        self.clearViewer()
        self.clearInputData()
        self.stitching_folder.setEnabled(True)
        self.file_button.setEnabled(True)
        self.reload_button.setEnabled(True)
        self.btnAbort.hide()
        self.info.set_output_summary('Aborted loading')
        if self._task is not None:
            self.cancel()
        self.finProgressBar()

    def cancel(self):
        """
        Cancel the current task, i.e. loading a file or sequence of files.
        """
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            # disconnect the `_task_finished` slot
            self._task.watcher.done.disconnect(self._task_finished)
            self._task = None

    # ----Qt Events----
    def keyPressEvent(self, event):
        if event.key() in (32, 16777220, 16777221):  # Enter keys and spacebar launch the reload option
            self.load_seq()
            self.return_pressed = True

    def onDeleteWidget(self):
        """Cancel running tasks like loading files before destroying widget"""
        self.cancel()
        super().onDeleteWidget()

    def displayDataIn(self):
        """Display the selected item in input tree viewer"""
        if not self.enable_multiselect:
            data = self.__treeViewerIn.get_selected_data()
            self.__dataViewer.setData(data)
        else:
            self.__dataViewer.setData(None)

    def displayDataOut(self):
        """Display the selected item in output tree viewer"""
        data = self.__treeViewerOut.get_selected_data()
        self.__dataViewer.setData(data)

    def enableOptions(self):
        """Enable checkboxes"""
        self.controls.enable_multiselect.setEnabled(True)

    def disableOptions(self):
        """Disable checkboxes"""
        self.controls.enable_multiselect.setEnabled(False)
        self.clearViewer()

    def clearViewer(self):
        """Clear data viewer and input file treeviewer"""
        self.__treeViewerIn.clearData()
        self.__dataViewer.setData(None)

    def close(self):
        # TODO: close any opened files
        super().close()

    def sizeHint(self):
        return QSize(1000, 500)

    @Inputs.data
    def set_data(self, data):
        """
        Linked to data input channel. Loads data from last widget as input data within this widget. The output data is not replaced.

        :param data: Input data
        :type data: dict
        """
        self.data_in = {}
        if data is not None:
            self.source = -1
            if isinstance(data, (tuple, np.ndarray, Table)):
                self.data_in = {'input_data': np.asarray(data)}
            elif isinstance(data, dict):
                self.data_in = data
            else:
                return self.Error.incorrect_input
            self.enableOptions()
        else:
            self.disableOptions()
            self.clear_messages()
        self.__treeViewerIn.updateDictionary(self.data_in)

    def select_file(self, n):
        """
        Select file from dropdown

        :param n: selected file number
        :type n: int
        """
        assert n < len(self.recent_paths)
        super().select_file(n)
        if self.recent_paths:
            self.load_data()
            self.set_file_list()

    def doneSelection(self):
        """
        Called after selected custom datasets.
        """
        self.done_custom_select.hide()
        selected = self.__treeViewerOut.selectedItems()
        selected_txt = [item.text(0) for item in selected
                        if (isinstance(self.data_out[item.text(0)], np.ndarray)
                            and item.text(0) not in self.DEFAULT_DATA_NAMES)]
        self.infoCustomData.setText('Custom datasets: ' + ', '.join(selected_txt))
        tmp = self.data_in['CUSTOM_DATA_NAMES'] if 'CUSTOM_DATA_NAMES' in self.data_in else []
        self.data_out['CUSTOM_DATA_NAMES'] = tmp + selected_txt
        self.__treeViewerOut.setSelectionMode(QAbstractItemView.SingleSelection)
        self.__treeViewerOut.clearSelection()
        if any(selected_txt):
            self.Outputs.data.send(self.data_out)

    def select_custom_data(self):
        """Enable selection of custom data"""
        self.done_custom_select.show()
        self.__treeViewerOut.setSelectionMode(QAbstractItemView.MultiSelection)

    def rightClickTreeItem(self, point):
        """
        Right click menu for file treeviewer

        :param point: xy position
        :type point: QPoint
        """
        item = self.__treeViewerOut.itemAt(point)
        self.selData = self.__treeViewerOut.get_selected_data(baseNode=item)
        if type(self.selData) in (np.ndarray, Quantity):
            self.menuMetData.exec_(self.__treeViewerOut.mapToGlobal(point))

    def convToMetrologyData(self):
        """Convert loaded data into orange-pylost data format MetrologyData"""
        self.source = 4
        kwargs = {}
        kwargs['unit'] = str(self.selData.unit) if type(self.selData) is Quantity else ''
        for item in ['pix_size', 'axis_names', 'detector_dimensions', 'full_size', 'start_pos']:
            if item in self.data_out:
                kwargs[item] = self.data_out[item]
        if 'full_size' not in kwargs and any(set(self.DEFAULT_DATA_NAMES).intersection(self.data_out.keys())):
            key = list(set(self.DEFAULT_DATA_NAMES).intersection(self.data_out.keys()))[0]
            kwargs['full_size'] = self.data_out[key].shape

        dialog = self.ConvertToMetrologyData(parent=self, **kwargs)
        if dialog.exec_():
            vals = dialog.get_all_params()
            self.update_metrology_data(vals, selItem=self.__treeViewerOut.currentItem())
            self.update_comment()
            self.__treeViewerOut.updateDictionary(self.data_out)
            self.Outputs.data.send(self.data_out)
            self.info.set_output_summary('Updated output data')
            # return self.Information.info('Updated output data')

    # def additionalData(self):
    #     self.source = 3
    #     kwargs = {}
    #     for item in ['pix_size', 'axis_names', 'detector_dimensions', 'full_size', 'start_pos']:
    #         if item in self.data_out:
    #             kwargs[item] = self.data_out[item]
    #     if 'full_size' not in kwargs and any(set(self.DEFAULT_DATA_NAMES).intersection(self.data_out.keys())):
    #         key = list(set(self.DEFAULT_DATA_NAMES).intersection(self.data_out.keys()))[0]
    #         kwargs['full_size'] = self.data_out[key].shape
    #
    #     dialog = self.ImportAdditionalDialog(parent=self, **kwargs)
    #     if dialog.exec_():
    #         vals = dialog.get_all_params()
    #         for item in vals:
    #             self.data_out[item] = vals[item]
    #
    #         self.update_metrology_data(vals)
    #         self.update_comment()
    #         self.__treeViewerOut.updateDictionary(self.data_out)
    #         self.Outputs.data.send(self.data_out)
    #         self.info.set_output_summary('Updated output data')
    #         # return self.Information.info('Updated output data')

    def clearOutput(self):
        """Clear output"""
        self.clearOutputData()
        self.Outputs.data.send(None)

    def clearOutputData(self):
        """Clear output file viewer and data viewer"""
        self.infoCustomData.setText('')
        self.__treeViewerOut.clearData()
        self.__dataViewer.setData(None)
        self.data_out = {}

    def clearInputData(self):
        """Clear input file viewer and data viewer"""
        self.__treeViewerIn.clearData()
        self.__dataViewer.setData(None)
        self.data_in = {}

    def update_comment(self):
        """Update comment log"""
        cmt = str(self.data_out['comment_log']) + '\n' if 'comment_log' in self.data_out else ''
        if cmt == '':
            cmt = self.data_in['comment_log'] + '\n' if 'comment_log' in self.data_in else ''
        self.data_out['comment_log'] = cmt
        if self.source == 1:
            self.data_out['comment_log'] = cmt + 'Data loaded from file {}.'.format(self.last_path())
            self.data_out['filename'] = os.path.basename(self.last_path())
        elif self.source == 2 and any(self.filename_seq):
            self.data_out['comment_log'] = cmt + 'Data loaded from file sequence [{}...]'.format(self.filename_seq[0])
            self.data_out['filename'] = os.path.basename(self.filename_seq[0])
        elif self.source == 3:
            self.data_out['comment_log'] = cmt + 'Additional data loaded from manual input.'
        # elif self.source==4:
        #     self.data_out['comment_log'] = cmt + 'Data converted to metrology datasets.'
        if 'h5path' in self.data_out:
            self.data_out['filename'] = '{}::{}'.format(self.data_out['filename'], self.data_out['h5path'])

    def sendOutputAll(self):
        """Send all the items in input to output"""
        self.clear_messages()
        if any(self.data_out):
            val = questionMsgAdv(title='Merge data?', msg='Output has existing data. Press "yes" to merge items, '
                                                          'and "no" to fully replace output data')
            if val == 1:
                self.data_out = {}
            elif val == 0:
                return
        copy_items(self.data_in, self.data_out, deepcopy=True)

        self.update_comment()
        self.__treeViewerOut.updateDictionary(self.data_out)
        self.Outputs.data.send(self.data_out)
        self.info.set_output_summary('Updated output data')
        # return self.Information.info('Updated output data')

    def sendOutput(self):
        """Send selected items in input to output"""
        self.clear_messages()
        selected = self.__treeViewerIn.selectedItems()
        if any(selected):
            selected_txt = [item.text(0) for item in selected]
            dialog = self.ImportDialog(parent=self, sel_items=selected_txt)
            if dialog.exec_():
                if any(self.data_out):
                    val = questionMsgAdv(title='Merge data?', msg='Press "yes" to only update / merge selected items, '
                                                                  'and "no" to fully replace output data')
                    if val == 1:
                        self.data_out = {}
                    elif val == 0:
                        return
                sel_keys = []
                for item in selected:
                    data = self.__treeViewerIn.get_selected_data(baseNode=item)
                    key = item.text(0)
                    unit = None
                    if key in dialog.map_items:
                        (key, unitWidget) = dialog.map_items[key]
                        unit = unitWidget.text()
                    if unit is not None and unit != '' and type(data) is not MetrologyData:
                        data = Quantity(data, unit=unit)
                    self.update_output(key, data)
                    sel_keys.append(key)

                self.update_metrology_data(vals=self.data_out, new_keys=sel_keys)
                self.update_comment()
                self.__treeViewerOut.updateDictionary(self.data_out)
                self.Outputs.data.send(self.data_out)
                self.info.set_output_summary('Updated output data')

    def update_metrology_data(self, vals, new_keys=[], selItem=None):
        """
        Convert input numpy arrays of height/slopes_x/slopes_y to MetrologyData and update output data.

        :param vals: Output data dictionary
        :type vals: dict
        :param new_keys: New names for the selected input items
        :type new_keys: list
        :param selItem: Selected output item, if any
        :type selItem: QTreeWidgetItem
        """
        if selItem is not None:
            selData = self.__treeViewerOut.get_selected_data(selItem)
            path = self.__treeViewerOut.get_selected_path(selItem, [])
            data, hasChanged = self.toMetrologyData(selData, vals, new_keys=new_keys, item=selItem)
            if hasChanged:
                fill_dict(self.data_out, path, data)
        else:
            for item in self.DEFAULT_DATA_NAMES:
                out = walk_dict(item, self.data_out, path=[])
                for path in out:
                    data = get_dict_item(self.data_out, path)
                    data, hasChanged = self.toMetrologyData(data, vals, new_keys=new_keys, item=item)
                    if hasChanged:
                        fill_dict(self.data_out, path, data)
            out = walk_dict_by_type(MetrologyData, self.data_out, path=[])
            for path in out:
                data = get_dict_item(self.data_out, path)
                if 'pix_size' in vals:
                    data._set_pix_size(vals['pix_size'])
                if 'detector_dimensions' in vals:
                    data._set_dim_detector(vals['detector_dimensions'])
                if 'axis_names' in vals:
                    data._set_axis_names(vals['axis_names'])
                if 'full_size' in vals:
                    data._set_init_shape(vals['full_size'])

    def toMetrologyData(self, data, vals, new_keys=[], item=''):
        """
        Convert from numpy array / Quantity to MetrologyData or update the MetrologyData parameters.

        :param data: Height or slope item data
        :type data: numpy.ndarray / Quantity / MetrologyData
        :param vals: Output data dictionary
        :type vals: dict
        :param new_keys: New names for the selected input items
        :type new_keys: list
        :param item: Selected output item, if any
        :type item: QTreeWidgetItem
        :return: Converted dataset, is conversion successful?
        :rtype: MetrologyData, bool
        """
        hasChanged = False
        if vals.get('xyz_to_z', False) and isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[-1] == 3:
            data, px, py = self.xyz_to_z(data)
            pix_unit = vals.get('pix_unit', '')
            vals['pix_size'] = [Quantity(py, pix_unit), Quantity(px, pix_unit)]
        if type(data) is np.ndarray and 'unit' in vals:
            data = Quantity(data, unit=vals['unit'])
            hasChanged = True
        if type(data) is Quantity and 'pix_size' in vals:
            pix_sz = vals['pix_size']
            dim_det = vals['det_dim'] if 'det_dim' in vals else self.default_dim_detector
            axis_names = vals['axis_names'] if 'axis_names' in vals else self.default_axis_names
            data = MetrologyData(data, pix_size=pix_sz, dim_detector=dim_det, axis_names=axis_names)
            hasChanged = True
        if type(data) is MetrologyData:
            motors = data.motors
            hasChangedMotors = False
            for key in ['motor_X', 'motor_Y', 'motor_Z', 'motor_RX', 'motor_RY', 'motor_RZ']:
                if key in vals and key in new_keys and isinstance(vals[key], Quantity):
                    mkey = {'name': key, 'values': vals[key].value.ravel(), 'axis': [], 'unit': str(vals[key].unit)}
                    if len(mkey['values']) not in data.shape:
                        self.Error.unknown(
                            '"{}" size ({}) does not find match in "{}" shape ({})'.format(key, len(mkey['values']),
                                                                                           item, data.shape))
                    else:
                        mkey['axis'] = data.shape.index(len(mkey['values']))
                        has_key, md = self.motors_hs_key(motors, key)
                        if has_key:
                            if isinstance(md['values'], (np.ndarray, list)) and len(md['values']) in data.shape:
                                if questionMsg(title='{} values exist'.format(key),
                                               msg='"{}" values already exist in "{}" data. Press "yes" to replace with new values'.format(
                                                   key, item)):
                                    motors = self.replace_motor_values(motors, key, mkey)
                                    hasChangedMotors = True
                            else:
                                # Replace motor values with new values
                                motors = self.replace_motor_values(motors, key, mkey)
                                hasChangedMotors = True
                        else:
                            motors = self.add_motor_values(motors, key, mkey)
                            hasChangedMotors = True
            if hasChangedMotors:
                hasChanged = True
                data._set_motors(motors)
        return data, hasChanged

    def xyz_to_z(self, data):
        """
        Convert data from xyz format to image format
        :param data: xyz data
        :type data: numpy.ndarray
        :return: image format, step x, step y
        :rtype: numpy.ndarray, float, float
        """
        x = np.unique(data[:, 0])
        y = np.unique(data[:, 1])
        dx = np.diff(x)
        dy = np.diff(y)
        if len(x) * len(y) == data.shape[0] and np.allclose(dx, dx[0]) and np.allclose(dy, dy[0]):
            return data[:, 2].reshape((len(y), len(x))), np.mean(dx), np.mean(dy)
        else:
            return data, 1.0, 1.0

    def add_motor_values(self, motors, key, mkey):
        """
        Add a new motor dataset

        :param motors: Motors list
        :type motors: list
        :param key: Name of the motor, e.g. motor_X
        :type key: str
        :param mkey: motor values, format {'values':[...], 'unit':'', 'name':''}
        :type mkey: dict
        :return: Updates motors
        :rtype: list
        """
        if motors is not None:
            motors.append(mkey)
        else:
            motors = [mkey]
        return motors

    def replace_motor_values(self, motors, key, mkey):
        """
        Replace motor values for given name

        :param motors: Motors list
        :type motors: list
        :param key: Name of the motor, e.g. motor_X
        :type key: str
        :param mkey: motor values, format {'values':[...], 'unit':'', 'name':''}
        :type mkey: dict
        :return: Updates motors
        :rtype: list
        """
        if motors is not None:
            for i, m in enumerate(motors):
                if m['name'] == key:
                    del motors[i]
                    motors.insert(i, mkey)
        return motors

    def motors_hs_key(self, motors, key):
        """
        Check if motor-list has the selected motor.

        :param motors: Motors list
        :type motors: list
        :param key: Name of the motor, e.g. motor_X
        :type key: str
        :return: Has motor?, values for selected motor
        :rtype: bool, dict
        """
        if motors is not None:
            for m in motors:
                if m['name'] == key:
                    return True, m
        return False, None

    def update_output(self, key, data):
        """
        Update output data dictionary with new key and values

        :param key: New key
        :type key: str
        :param data: New value for the key
        :type data:
        """
        if len(data) == 1:
            data = data[0]
            self.source = 1
        out = walk_dict(key, self.data_out, path=[])
        if not any(out):
            self.data_out[key] = data
        elif len(out) == 1:
            fill_dict(self.data_out, out[0], data)
        elif len(out) > 1:
            val = questionMsgAdv(title='Update multiple items',
                                 msg='Many items found with selected key "{}". '
                                     'Press "yes" to update all items, and "no" to update only first found item'.format(key))
            if val == 2:
                for path in out:
                    fill_dict(self.data_out, path, data)
            elif val == 1:
                fill_dict(self.data_out, out[0], data)
        self.tabs.setCurrentIndex(1)

    def browse_file_seq(self):
        """Load a sequence of files, e.g. subapertures"""
        try:
            start_file = self.last_path() or os.path.expanduser("~/")

            readers = [f for f in FileFormat.formats
                       if getattr(f, 'read', None)
                       and getattr(f, "EXTENSIONS", None)]
            self.filename_list, reader, _ = open_multifile_dialog(start_file, None, readers)
            if reader is not None:
                self.recent_paths[0].file_format = reader.qualified_name()
            self.load_seq()
        except Exception as ex:
            log.exception(ex)

    def load_stitching_root(self):
        try:
            start_file = self.last_path() or os.path.expanduser("~/")

            readers = [f for f in FileFormat.formats
                       if getattr(f, 'read', None)
                       and getattr(f, "EXTENSIONS", None)]
            filename, reader, _ = open_filename_dialog(start_file, None, readers)
            if filename is None:
                return
            if reader is not None:
                self.recent_paths[0].file_format = reader.qualified_name()
            filename = Path(filename)
            root_folder = filename.parent
            start = re.search(r'\d+$', filename.stem).start()
            root_name = filename.stem[:start]
            lis = root_folder.glob(root_name + '*' + filename.suffix)
            filename_list = []
            idx = []
            for f in lis:
                if len(f.stem) > len(filename.stem) + 3:  # assumption
                    continue
                filename_list.append(str(f))
                idx.append(int(re.search(r'\d+$', f.stem).group()))
            idx = np.asarray(idx) - min(idx)
            self.filename_list = list(np.asarray(filename_list)[np.argsort(idx)])
            self.load_seq()
        except Exception as ex:
            log.exception(ex)

    def load_seq(self):
        if self.return_pressed == True:
            return
        try:
            if self.filename_list is None:
                return
            if len(self.filename_list) == 1:
                self.add_path(self.filename_list[0])
                self.load_data()
                return
            if any(self.filename_list):
                if self._task is not None:
                    self.cancel()
                assert self._task is None

                self.setStatusMessage('')
                self.clear_messages()
                self.clearViewer()
                self.filename_seq = self.filename_list
                self.add_path(self.filename_list[0])
                self.data_in = {}
                self.reader = FileFormat.get_reader(
                    self.filename_list[0])  # if reader is None else reader(self.filename_list[0])

                seq_cls = FileSeqLoader()
                seq_cls.progress.connect(self.report_progress)
                self._task = task = Task()
                end_progressbar = methodinvoke(self, "finProgressBar", ())

                def callback():
                    if task.cancelled:
                        end_progressbar()
                        raise Exception('Aborted loading')

                load_fun = partial(seq_cls.load_file_seq, callback=callback, filename_list=self.filename_list,
                                   reader=self.reader)

                self.startProgressBar()
                task.future = self._executor.submit(load_fun)
                task.watcher = FutureWatcher(task.future)
                task.watcher.done.connect(self._task_finished)
                self.stitching_folder.setEnabled(False)
                self.reload_button.setEnabled(False)
                self.file_button.setEnabled(False)
                self.btnAbort.show()
        except Exception as ex:
            log.exception(ex)
            try:
                self.progressBarFinished()
            except Exception as e:
                pass
            return lambda x=ex: self.Error.unknown(repr(x))

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        """
        Callback after the file sequence is completed.

        :param f: Result containing loaded file sequence data
        :type f: concurrent.futures.Future
        """
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None
        self.endProgressBar()
        try:
            self.stitching_folder.setEnabled(True)
            self.file_button.setEnabled(True)
            self.reload_button.setEnabled(True)
            self.btnAbort.hide()

            self.data_in = f.result()
            self.source = 2
            if any(self.data_in):
                self.autoFillOutput(self.reader)
                self.__treeViewerIn.updateDictionary(self.data_in)
                self.setStatusMessage('{}'.format(os.path.basename(self.filename_seq[0])))
                self.return_pressed == False
                if config_params.DEFAULT_CLOSE_WIDGETS_AFTER_APPLY:
                    self.close()
        except Exception as e:
            return self.Error.unknown(repr(e))

    def report_progress(self, val):
        """
        Update progressbar

        :param val: New progress value
        :type val: float
        """
        try:
            self.setProgressValue(val)
        except Exception as e:
            self.Error.unknown(repr(e))

    @pyqtSlot(float)
    def setProgressValue(self, value):
        """
        Set progressbar value

        :param value: New progress value
        :type value: float
        """
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    @pyqtSlot()
    def finProgressBar(self):
        """
        Finish progress bar
        """
        assert self.thread() is QThread.currentThread()
        self.endProgressBar()

    def startProgressBar(self):
        """Start progress bar"""
        try:
            self.progressBarInit()
        except Exception as e:
            self.Error.unknown(repr(e))

    def endProgressBar(self):
        """End progress bar"""
        try:
            self.progressBarFinished()
        except Exception as e:
            self.Error.unknown(repr(e))

    def browse_file(self):
        """Browse a single file with known extension, and load data"""
        start_file = self.last_path() or os.path.expanduser("~/")

        readers = [f for f in FileFormat.formats
                   if getattr(f, 'read', None)
                   and getattr(f, "EXTENSIONS", None)]
        filename, reader, _ = open_filename_dialog(start_file, None, readers)
        # filename, filter = QFileDialog.getOpenFileName(self, "Select File", start_file,'*.h5')
        if not filename:
            return
        self.add_path(filename)
        if reader is not None:
            self.recent_paths[0].file_format = reader.qualified_name()

        self.load_data()

    # Open a file, create data from it and send it over the data channel
    def load_data(self):
        """Load data from a file"""
        # We need to catch any exception type since anything can happen in
        # file readers
        self.setStatusMessage('')
        self.clear_messages()
        self.set_file_list()

        if self.last_path() and not os.path.exists(self.last_path()):
            return self.Error.file_not_found

        error = self._try_load()
        if error:
            error()
            self.data_out = {}
            self.Outputs.data.send(None)
            self.infolabel.setText("No data.")
            self.info.set_output_summary(self.info.NoOutput)

    def nslsmetadata(self, data):
        try:
            fileseq = []
            motorX = []
            motorY = []
            h5file = self.filename_list[0]
            wdir = Path(h5file).parent
            for key, val in data.items():
                if isinstance(val, np.ndarray):
                    header = val[0]
                    if len(header) > 5:
                        return data
                    for idx, column in enumerate(header):
                        if 'Filename' in str(column):
                            filename_list = np.asarray(val[1:, idx], dtype=str)
                            self.filename_list = []
                            for file in filename_list:
                                self.filename_list.append(str(Path(wdir, file)))
                        elif 'X Location' in str(column):
                            unitX = val[0, idx].decode().split('[')[-1].split(']')[0]
                            motorX = Quantity(np.asarray(val[1:, idx], dtype=float), unitX).to('mm')
                        elif 'Y Location' in str(column):
                            unitY = val[0, idx].decode().split('[')[-1].split(']')[0]
                            motorY = Quantity(np.asarray(val[1:, idx], dtype=float), unitY).to('mm')

            # self.filename_list = self.filename_list[:4]  ####

            try:
                motors = self.data_out['height'].motors
            except:
                motors = [{'name': 'motor_X', 'values': None, 'unit': None},
                          {'name': 'motor_Y', 'values': None, 'unit': None}]
            for motor in motors:
                if 'motor_X' in motor['name']:
                    self.data_in['motorX'] = motorX.value#[:4]  #test
                    motor['values'] = motorX.value#[:4]  #test
                    motor['unit'] = 'mm'
                    motor['axis'] = -1
                elif 'motor_Y' in motor['name']:
                    self.data_in['motorY'] = motorY.value#[:4]  #test
                    motor['values'] = motorY.value#[:4]  #test
                    motor['unit'] = 'mm'
                    motor['axis'] = -2
            self.updatemotors = motors
            self.load_seq()
            self.filename_list = [h5file, ]
            return self.data_in
        except:
            return data

    def _try_load(self):
        """Call file reader 'read' function to load data"""
        try:
            self.reader = self._get_reader()
            assert self.reader is not None
        except Exception:
            return self.Error.missing_reader

        if self.reader is self.NoFileSelected:
            self.Outputs.data.send(None)
            self.info.set_output_summary(self.info.NoOutput)
            return None

        with catch_warnings(record=True) as warnings:
            try:
                data = self.reader.read()
                self.clearViewer()
                self.updatemotors = None
                if type(data) is dict:
                    # convert nsls h5 metadata file if necessary
                    self.data_in = self.nslsmetadata(data)
                else:
                    self.data_in = {'file_data': np.array(data) if type(data) is Table else data}

                if len(self.data_in) < 2:  # ndarray typically
                    for entry in self.data_in.values():
                        if isinstance(entry, np.ndarray):
                            height = MetrologyData(entry, unit='nm', pix_size=1, pix_unit='mm',
                                                   dim_detector=[-3, -2, -1], axis_names=['Z', 'Y', 'X'],)
                            self.data_in = {'height' : height}

                self.source = 1
                if self.updatemotors is None:
                    self.__treeViewerIn.updateDictionary(self.data_in)
                    self.autoFillOutput(self.reader)
                if 'h5path' in self.data_in:
                    self.setStatusMessage('{}::{}'.format(os.path.basename(self.last_path()), self.data_in['h5path']))
                else:
                    self.setStatusMessage('{}'.format(os.path.basename(self.last_path())))
                if config_params.DEFAULT_CLOSE_WIDGETS_AFTER_APPLY:
                    self.close()
            except Exception as ex:
                log.exception(ex)
                self.setStatusMessage('')
                return lambda x=ex: self.Error.unknown(str(x))
            if warnings:
                self.Warning.load_warning(warnings[-1].message.args[0])

        self.infolabel.setText("Data loaded.")
        return None

    def _get_reader(self) -> FileFormat:
        """Get file reader"""
        path = self.last_path()
        if path is None:
            return self.NoFileSelected
        if self.recent_paths and self.recent_paths[0].file_format:
            qname = self.recent_paths[0].file_format
            reader_class = class_from_qualified_name(qname)
            reader = reader_class(path)
        else:
            reader = FileFormat.get_reader(path)
        if self.recent_paths and self.recent_paths[0].sheet:
            reader.select_sheet(self.recent_paths[0].sheet)
        return reader

    def check_multiselect(self):
        """Check multiple selection in the treeview"""
        if self.enable_multiselect:
            self.__treeViewerIn.setSelectionMode(QAbstractItemView.MultiSelection)
        else:
            self.__treeViewerIn.setSelectionMode(QAbstractItemView.SingleSelection)

    # TODO : Correctly specify which axis is represented by which motor
    def autoFillOutput(self, reader):
        """Auto fill output if implemented by the file reader in 'data_standard_format' function"""
        data = None
        motors = None
        if getattr(reader, 'clear_output_before_loading', False):
            self.clearOutputData()
        if type(reader) in [PickleReader, H5Reader]:
            data = self.data_in
            copy_items(data, self.data_out, deepcopy=True)
        elif hasattr(reader, 'data_standard_format'):
            data = reader.data_standard_format(self.data_in)
            for key in data:
                self.update_output(key, data[key])
        else:
            self.tabs.setCurrentIndex(0)

        if data is not None:
            # self.update_metrology_data(vals=self.data_out)
            for item in data.values():
                if isinstance(item, MetrologyData):
                    self.pixel_size = self._format_vis(item.pix_size_detector, fmt='{:.6f}')
                    break
            self.update_comment()
            data['module'] = 'custom'
            if self.updatemotors is not None:
                motors = self.updatemotors
                for motor in motors:
                    if 'motor_X' in motor['name']:
                        self.data_in['motorX'] = motor['values']
                    elif 'motor_Y' in motor['name']:
                        self.data_in['motorY'] = motor['values']
                self.__treeViewerIn.updateDictionary(self.data_in)
                self.data_out['height'].update_motors(motors)
            self.__treeViewerOut.updateDictionary(self.data_out)
            self.Outputs.data.send(self.data_out)
            self.info.set_output_summary('Updated output data')

    class ConvertToMetrologyData(QDialog, UI_met_data):
        """Convert a numpy array or astropy.Quantity into MEtrologyData"""

        def __init__(self, parent=None, **kwargs):
            QDialog.__init__(self, parent)
            self.setupUi(self)

            if 'unit' in kwargs:
                self.dataset_unit.setText(self._format_vis(kwargs['unit']))
            if 'pix_size' in kwargs:
                if isinstance(kwargs['pix_size'], Quantity):
                    self.pix_size.setText(self._format_vis(kwargs['pix_size'].value))
                    self.pix_size_unit.setText(self._format_vis(kwargs['pix_size'].unit))
                else:
                    self.pix_size.setText(self._format_vis(kwargs['pix_size']))
            if 'axis_names' in kwargs:
                self.axis_names.setText(self._format_vis(kwargs['axis_names']))
            if 'detector_dimensions' in kwargs:
                self.det_dim.setText(self._format_vis(kwargs['detector_dimensions']))
            if 'full_size' in kwargs:
                self.full_size.setText(self._format_vis(kwargs['full_size']))
            if 'start_pos' in kwargs:
                self.start_pos.setText(self._format_vis(kwargs['start_pos']))

            self.buttonBox.accepted.connect(self.accept)
            self.buttonBox.rejected.connect(self.reject)

        def get_all_params(self):
            """Get all parameters entered in the UI"""
            out = {'unit': self.dataset_unit.text()}
            val_unit = self.pix_size_unit.text()
            out['pix_unit'] = val_unit
            val = self._parse_data(self.pix_size.text())
            if val is not None:
                out['pix_size'] = Quantity(val, unit=val_unit)
            val = self._parse_str(self.axis_names.text())
            if val is not None:
                out['axis_names'] = val
            val = self._parse_data(self.det_dim.text())
            if val is not None:
                out['detector_dimensions'] = val
            val = self._parse_data(self.full_size.text())
            if val is not None:
                out['full_size'] = val
            val = self._parse_data(self.start_pos.text())
            if val is not None:
                out['start_pos'] = val
            out['xyz_to_z'] = self.xyz_to_z.isChecked()
            return out

        def _format_vis(self, val):
            """
            Format visulization of params in the UI

            :param val: Param value
            :type val: int/float/str
            :return: Formatted value as string
            :rtype: str
            """
            if isinstance(val, (tuple, list, np.ndarray)):
                return ' x '.join(str(x) for x in val)
            else:
                return '{}'.format(val)

        def _parse_data(self, val):
            """
            Parse entered parameter in string format.

            :param val: PAram value in UI
            :type val: str
            :return: Parse value
            :rtype: int/float/str/bool
            """
            try:
                if ' x ' in val:
                    return [ast.literal_eval(x.strip()) for x in val.split(' x ')]
                else:
                    return ast.literal_eval(val)
            except Exception:
                return None

        def _parse_str(self, val):
            """
            Parse string stripping edges

            :param val: Input string
            :type val: str
            :return: Parsed string
            :rtype: str
            """
            try:
                if ' x ' in val:
                    return [x.strip() for x in val.split(' x ')]
                elif val.strip() == '':
                    return None
                else:
                    return val
            except Exception:
                return None

    class ImportAdditionalDialog(QDialog, UI_import_addnl):
        """Not used"""

        def __init__(self, parent=None, **kwargs):
            QDialog.__init__(self, parent)
            self.setupUi(self)

            if 'pix_size' in kwargs:
                if isinstance(kwargs['pix_size'], Quantity):
                    self.pix_size.setText(self._format_vis(kwargs['pix_size'].value))
                    self.pix_size_unit.setText(self._format_vis(kwargs['pix_size'].unit))
                else:
                    self.pix_size.setText(self._format_vis(kwargs['pix_size']))
            if 'axis_names' in kwargs:
                self.axis_names.setText(self._format_vis(kwargs['axis_names']))
            if 'detector_dimensions' in kwargs:
                self.det_dim.setText(self._format_vis(kwargs['detector_dimensions']))
            if 'full_size' in kwargs:
                self.full_size.setText(self._format_vis(kwargs['full_size']))
            if 'start_pos' in kwargs:
                self.start_pos.setText(self._format_vis(kwargs['start_pos']))

            self.buttonBox.accepted.connect(self.accept)
            self.buttonBox.rejected.connect(self.reject)

        def get_all_params(self):
            out = {}
            val = self._parse_data(self.pix_size.text())
            if val is not None:
                val_unit = self.pix_size_unit.text()
                out['pix_size'] = Quantity(val, unit=val_unit)
            val = self._parse_str(self.axis_names.text())
            if val is not None:
                out['axis_names'] = val
            val = self._parse_data(self.det_dim.text())
            if val is not None:
                out['detector_dimensions'] = val
            val = self._parse_data(self.full_size.text())
            if val is not None:
                out['full_size'] = val
            val = self._parse_data(self.start_pos.text())
            if val is not None:
                out['start_pos'] = val
            return out

        def _format_vis(self, val):
            if isinstance(val, (tuple, list, np.ndarray)):
                return ' x '.join(str(x) for x in val)
            else:
                return '{}'.format(val)

        def _parse_data(self, val):
            try:
                if ' x ' in val:
                    return [ast.literal_eval(x.strip()) for x in val.split(' x ')]
                else:
                    return ast.literal_eval(val)
            except Exception:
                return None

        def _parse_str(self, val):
            try:
                if ' x ' in val:
                    return [x.strip() for x in val.split(' x ')]
                elif val.strip() == '':
                    return None
                else:
                    return val
            except Exception:
                return None

    class ImportDialog(QDialog, UI_import):
        """Import data dialog"""

        def __init__(self, parent=None, sel_items=[]):
            QDialog.__init__(self, parent)
            self.setupUi(self)

            for i, item in enumerate(sel_items):
                lbl = gui.widgetLabel(None, item)
                lbl.setSizePolicy(Policy.Fixed, Policy.Fixed)
                self.grid_layout.addWidget(lbl, i, 0, alignment=Qt.AlignRight)
                btn = gui.button(None, self, 'O', callback=lambda obj, i=i: self.left_btn_click(i), autoDefault=False)
                btn.setMaximumWidth(30)
                btn.setSizePolicy(Policy.Fixed, Policy.Fixed)
                self.grid_layout.addWidget(btn, i, 1)
                self.grid_layout.addWidget(QLabel(''), i, 2, alignment=Qt.AlignCenter)

            IMPORT_NAMES = get_import_data_names()
            if len(IMPORT_NAMES) < len(sel_items):
                IMPORT_NAMES += [''] * (len(sel_items) - len(IMPORT_NAMES))
            for j, item in enumerate(IMPORT_NAMES):
                btn = gui.button(None, self, 'O', callback=lambda obj, j=j: self.right_btn_click(j), autoDefault=False)
                btn.setMaximumWidth(30)
                btn.setSizePolicy(Policy.Fixed, Policy.Fixed)
                self.grid_layout.addWidget(btn, j, 3)
                lbl = gui.widgetLabel(None, item)
                lbl.setSizePolicy(Policy.Fixed, Policy.Fixed)
                self.grid_layout.addWidget(lbl, j, 4, alignment=Qt.AlignLeft)
                le = QLineEdit()
                le.setPlaceholderText('Unit')
                le.setMaximumWidth(50)
                le.setSizePolicy(Policy.Fixed, Policy.Fixed)
                self.grid_layout.addWidget(le, j, 5, alignment=Qt.AlignLeft)

            self.grid_layout.addWidget(QLabel('              '), max(i, j) + 1, 2, alignment=Qt.AlignRight)

            self.buttonBox.accepted.connect(self.accept)
            self.buttonBox.rejected.connect(self.reject)

            self.startPos = None
            self.endPos = None
            self.sel_left_pos = -1
            self.sel_right_pos = -1
            self.map_items = {}
            self.setMouseTracking(True)

        def setMouseTracking(self, flag):
            """Link dataset names to standardized name set"""

            def recursive_set(parent):
                for child in parent.findChildren(QtCore.QObject):
                    try:
                        child.setMouseTracking(flag)
                    except:
                        pass
                    recursive_set(child)

            QtWidgets.QWidget.setMouseTracking(self, flag)
            recursive_set(self)

        def left_btn_click(self, i):
            """Start drawing line"""
            button = self.sender()
            self.startPos = button.pos() + QPoint(button.width(), button.height() / 2)
            self.sel_left_pos = i
            self.update()

        def right_btn_click(self, j):
            """End drawing line"""
            button = self.sender()
            self.sel_right_pos = j
            self.updateConnect()
            self.update()

        def mousePressEvent(self, event):
            """Update connections and nodes"""
            super(OWFileExternal.ImportDialog, self).mousePressEvent(event)
            self.updateConnect()
            self.update()

        def updateConnect(self):
            """REorder items based on connection"""
            try:
                self.startPos = None
                if self.sel_left_pos != -1:
                    if self.sel_right_pos != -1:
                        self.grid_layout.itemAtPosition(self.sel_left_pos, 1).widget().setText('X')
                        self.grid_layout.itemAtPosition(self.sel_left_pos, 3).widget().setText('X')
                        self.grid_layout.itemAtPosition(self.sel_left_pos, 2).widget().setText('-->')
                        unitWidget = self.grid_layout.itemAtPosition(self.sel_left_pos, 5).widget()
                        left = self.grid_layout.itemAtPosition(self.sel_left_pos, 0).widget()
                        right_new = self.grid_layout.itemAtPosition(self.sel_left_pos, 4).widget()
                        right_old = self.grid_layout.itemAtPosition(self.sel_right_pos, 4).widget()
                        temp = right_old.text()
                        right_old.setText(right_new.text())
                        right_new.setText(temp)
                        self.map_items[left.text()] = (temp, unitWidget)
                    else:
                        self.grid_layout.itemAtPosition(self.sel_left_pos, 1).widget().setText('O')
                        self.grid_layout.itemAtPosition(self.sel_left_pos, 3).widget().setText('O')
                        self.grid_layout.itemAtPosition(self.sel_left_pos, 2).widget().setText('')
                        left = self.grid_layout.itemAtPosition(self.sel_left_pos, 0).widget()
                        del self.map_items[left.text()]
                self.sel_left_pos = -1
                self.sel_right_pos = -1

            except Exception as e:
                print(e)

        def mouseMoveEvent(self, event):
            """Mouse move event:update line drawing"""
            super(OWFileExternal.ImportDialog, self).mouseMoveEvent(event)
            self.endPos = event.pos()
            self.update()

        def paintEvent(self, event):
            """Draw line"""
            QDialog.paintEvent(self, event)
            if self.startPos and self.endPos:
                painter = QPainter(self)
                pen = QPen(Qt.black, 3, Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(self.startPos.x(), self.startPos.y(), self.endPos.x(), self.endPos.y())
