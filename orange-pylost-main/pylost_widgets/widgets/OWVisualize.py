# coding=utf-8
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QSizePolicy, QTabWidget
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.util.util_functions import MODULE_MULTI, MODULE_SINGLE, copy_items
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWVisualize(PylostWidgets, PylostBase):
    name = 'Visualize'
    description = 'Plot raw/stitched data. Integrate / differentiate slopes and heights data.'
    icon = "../icons/vis.svg"
    priority = 24

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = False

    module = Setting('', schema_only=True)
    scan_name = Setting('')
    interpolate = Setting(True, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)
        self.has_updates = False

        box = super().init_info(module=True, module_callback=self.change_module, scans=True,
                                scans_callback=self.change_scan)
        self.btnReload = gui.button(box, self, 'Reload', callback=self.update_data,
                                    sizePolicy=(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.checkInterpolate = gui.checkBox(self.controlArea, self, 'interpolate',
                                             'Interpolate NaN values (for integration)', callback=self.change_interp)
        self.btnReload.hide()

        # Data viewer
        box = gui.vBox(self.controlArea, "Viewer", stretch=9)
        self.tabs = QTabWidget(self)
        self.dataViewers = {}
        box.layout().addWidget(self.tabs)

    def sizeHint(self):
        return QSize(1000, 750)

    @Inputs.data
    def set_data(self, data):
        self.data_in = {}
        self.data_out = {}
        self.clear_all()
        if data is not None:
            self.data_in = data
        self.has_updates = True
        self.btnReload.show()

    def activateWindow(self):
        self.update_data()
        super().activateWindow()

    def show(self):
        self.update_data()
        super().show()

    def focusInEvent(self, event):
        self.update_data()
        super().focusInEvent(event)

    def update_data(self):
        if self.has_updates:
            if any(self.data_in):
                copy_items(self.data_in, self.data_out, deepcopy=True)
                self.update_tabs(self.data_in, show_all_default_names=True)
                self.load_data()
            else:
                self.update_tabs(None)
                self.clear_messages()
            self.has_updates = False
            self.btnReload.hide()

    def clear_all(self):
        self.selModule.clear()
        self.selScan.clear()
        self.clear_viewers()

    def load_data(self, multi=False):
        super().load_data()
        self.change_module()

    def change_module(self):
        self.info.set_output_summary('Loading...')
        QtWidgets.qApp.processEvents()
        self.selScan.setEnabled(False)
        self.selScan.clear()
        self.clear_viewers()
        module_data = self.get_data_by_module(self.data_out, self.module)
        if self.module in MODULE_MULTI:
            if len(module_data) > 0:
                self.selScan.setEnabled(True)
                self.selScan.addItems(list(module_data.keys()))
                self.change_scan()
        elif self.module in MODULE_SINGLE:
            scan_vis = self.full_dataset(module_data)
            self.load_viewer(scan_vis)
            self.set_data_by_module(self.data_out, self.module, scan_vis)
            self.Outputs.data.send(self.data_out)
        self.info.set_output_summary('')

    def change_scan(self):
        curScan = self.selScan.currentText()
        scans = self.get_data_by_module(self.data_out, self.module)
        scans_vis = self.load_full_dataset(scans, curScan)
        self.load_viewer(scans_vis[curScan])

    def change_interp(self):
        curScan = self.selScan.currentText()
        scans = self.get_data_by_module(self.data_in, self.module)
        scans_vis = self.load_full_dataset(scans, curScan)
        self.load_viewer(scans_vis[curScan])

    def load_full_dataset(self, scans, curScan):
        scans_vis = {}
        scan_vis = self.full_dataset(scans[curScan])
        for item in scans:
            scans_vis[item] = scans[item] if item != curScan else scan_vis
        # scans_out = self.get_data_by_module(self.data_out, self.module)
        # scans_out = scans_vis
        self.set_data_by_module(self.data_out, self.module, scans_vis)
        self.Outputs.data.send(self.data_out)
        return scans_vis
