# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy as Policy, QTabWidget
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import MODULE_MULTI, MODULE_SINGLE, copy_items, parseQInt
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWSelectSubapertures(PylostWidgets, PylostBase):
    name = 'Select Subapertures'
    description = 'Select subapertures widget.'
    icon = "../icons/filter-subap.svg"
    priority = 61

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0

    module = Setting('', schema_only=True)
    scan_name = Setting('')
    exclude_items = Setting('')
    keep_excludes = Setting(False)

    ALL_SCANS, CUR_SCAN, ENTER_SCANS = range(3)
    scan_option = Setting(ALL_SCANS, schema_only=True)
    apply_scans_txt = Setting('')

    class Information(widget.OWWidget.Information):
        info = Msg("Info:\n{}")

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)
        self.shp_motor = []
        self.nb_subap = 0
        self.sel_subap = {}
        self.sel_subap_vis = {}
        self.qexcl_arr = []
        self.qscans_arr = []

        box = super().init_info(module=True, module_callback=self.change_module, scans=True,
                                scans_callback=self.change_scan)
        self.btnApply = gui.button(box, self, 'Apply selection', callback=self.apply_selection, autoDefault=False,
                                   stretch=1, sizePolicy=Policy(Policy.Fixed, Policy.Fixed))

        self.boxScan = gui.vBox(self.controlArea, "Scan Options", stretch=1)
        rbox = gui.radioButtons(self.boxScan, self, "scan_option", box=False, addSpace=True)
        gui.appendRadioButton(rbox, 'Apply selection for all scans')
        gui.appendRadioButton(rbox, 'Apply selection for current scan')
        gui.appendRadioButton(rbox, 'Apply selection for following scans')
        forScans = gui.lineEdit(self.boxScan, self, 'apply_scans_txt', label='Enter scans', orientation=Qt.Horizontal)
        forScans.setPlaceholderText('e.g. "0-5, 8, 10" and press enter')

        box = gui.vBox(self.controlArea, "Options", stretch=1)
        exclSubap = gui.lineEdit(box, self, 'exclude_items', label='Exclude subapertures',
                                 callback=self.apply_selection, orientation=Qt.Horizontal, addSpace=True)
        exclSubap.setPlaceholderText(
            'e.g. "0-10, 15; 1,2,3" and press enter. ";" for dimension / axis, "," for subapertures')
        gui.checkBox(box, self, 'keep_excludes', 'Keep excluded subapertures and fill them with NaN', addSpace=True)
        self.lblOrigSz = gui.label(box, self, '')
        self.lblOrigSz.setStyleSheet('color: red')

        # Data viewer
        box1 = gui.vBox(self.controlArea, "Data viewer", stretch=19)
        self.tabs = QTabWidget(self)
        self.dataViewers = {}
        box1.layout().addWidget(self.tabs)

    def sizeHint(self):
        return QSize(700, 500)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_tabs=True)
        if data is None:
            self.Outputs.data.send(None)
            self.selScan.clear()

    def load_data(self, multi=False):
        super().load_data()
        self.change_module()
        self.Outputs.data.send(self.data_out)

    def change_module(self):
        self.selScan.clear()
        self.clear_viewers()
        module_data = self.get_data_by_module(self.data_in, self.module)
        if self.module in MODULE_MULTI:
            self.boxScan.show()
            self.selScan.parent().show()
            if len(module_data) > 0:
                self.selScan.setEnabled(True)
                self.selScan.addItems(list(module_data.keys()))
        else:
            self.boxScan.hide()
            self.selScan.parent().hide()
        self.apply_selection()

    def change_scan(self):
        self.update_viewer()

    def update_comment(self, comment='', prefix=''):
        super().update_comment(comment, prefix='Select subapertures')

    def update_viewer(self):
        module_data = self.get_data_by_module(self.data_out, self.module)
        if self.module in MODULE_MULTI:
            curScan = self.selScan.currentText()
            self.load_viewer(module_data[curScan], show_mask=False)
        elif self.module in MODULE_SINGLE:
            self.load_viewer(module_data, show_mask=False)

    def parse_excl_txt(self):
        excl_arr = []
        if self.controls.exclude_items.text() != '':
            txt = self.controls.exclude_items.text()
            txt_arr = txt.split(';')
            for txt_i in txt_arr:
                excl_arr.append(parseQInt(txt_i))
        return excl_arr

    def get_scans_comment(self):
        if self.scan_option == self.ALL_SCANS:
            return 'for all scans'
        elif self.scan_option == self.CUR_SCAN:
            return 'for "{}"'.format(self.selScan.currentText())
        elif self.scan_option == self.ENTER_SCANS:
            return 'for {} scans'.format(self.scan_arr)

    def apply_selection(self):
        self.excl_arr = []
        self.scan_arr = parseQInt(self.apply_scans_txt) if self.apply_scans_txt != '' else []
        copy_items(self.data_in, self.data_out)
        if self.exclude_items != '':
            self.excl_arr = self.parse_excl_txt()
            super().apply_scans()
        self.Outputs.data.send(self.data_out)
        self.update_viewer()

    def apply_scan(self, scan, scan_name=None, comment=''):
        if scan_name is not None:
            if self.scan_option == self.CUR_SCAN and scan_name != self.selScan.currentText():
                return scan, comment
            elif self.scan_option == self.ENTER_SCANS and len(self.scan_arr) > 0 and int(
                    scan_name.split('_')[1]) not in self.scan_arr:
                return scan, comment
        return super().apply_scan(scan, scan_name, comment)

    def apply_scan_item(self, Z, comment='', item=None):
        dims = super().get_detector_dimensions(Z)
        dims_m = np.invert(dims)
        axes_m = dims_m.nonzero()[0]
        shp_m = np.array(Z.shape)[dims_m]

        slc = [slice(None)] * Z.ndim
        for i in range(len(self.excl_arr)):
            if i < len(shp_m):
                slc[axes_m[i]] = [x for x in np.arange(shp_m[i]) if x not in self.excl_arr[i]]
        if self.keep_excludes:
            tmp = np.full_like(Z, np.nan)
            tmp[slc] = Z[slc]
            Zret = tmp
        else:
            Zret = Z[slc]
            if isinstance(Z, MetrologyData):
                self.update_motors(Z, Zret, slc)
        self.lblOrigSz.setText('Original subapertures shape {}'.format(tuple(shp_m)))
        if self.module in MODULE_MULTI:
            comment = 'subapertures {} are {}, {}.'.format(self.exclude_items,
                                                           'set to NaN' if self.keep_excludes else 'removed',
                                                           self.get_scans_comment())
        else:
            comment = 'subapertures {} are {}.'.format(self.exclude_items,
                                                       'set to NaN' if self.keep_excludes else 'removed')
        return Zret, comment

    def update_motors(self, scan_item, scan_out_item, mask_list):
        try:
            if any(scan_item._motors):
                for i, val in enumerate(scan_item.dim_detector):
                    for j in range(len(scan_item._motors)):
                        m = scan_item._motors[j]
                        if not val and 'values' in m and scan_item.shape[i] == len(m['values']):
                            scan_out_item._motors[j]['values'] = m['values'][mask_list[i]]

        except Exception as e:
            print(e)
            self.Error.unknown(repr(e))
