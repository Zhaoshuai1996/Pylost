# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QSizePolicy as Policy, QTabWidget
from astropy import units as u
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.math import pv, rms
from pylost_widgets.util.util_functions import MODULE_MULTI, MODULE_SINGLE, copy_items, fit_nD_metrology
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWStats(PylostWidgets, PylostBase):
    name = 'Statistics'
    description = 'Plot tilt, radius and figure/slope error rms for a sequence of images.'
    icon = "../icons/statistics.svg"
    priority = 84

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    scan_name = Setting('', schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True, module_callback=self.change_module, scans=True, scans_callback=self.apply)
        self.btnApply = gui.button(box, self, 'Apply', callback=self.apply, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        # Data viewer
        box = gui.vBox(self.controlArea, "Viewer", stretch=9)
        self.tabs = QTabWidget(self)
        self.figure = {}
        self.canvas = {}
        self.toolbar = {}
        box.layout().addWidget(self.tabs)

    def sizeHint(self):
        return QSize(800, 900)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_tabs=True, show_only_default=True)

    def load_data(self, multi=False):
        super().load_data()
        self.change_module()

    def add_new_tab(self, name):
        """
        Add a new tab in the TabWidget with the given name. The tab by default has only a silx DataViewer

        :param name: Name of the tab
        :type name: str
        """

        self.figure[name] = Figure(figsize=(6.32, 6.32), edgecolor='gray', linewidth=0.1, tight_layout=True)
        self.canvas[name] = FigureCanvas(self.figure[name])
        self.toolbar[name] = NavigationToolbar(self.canvas[name], self)
        box = gui.vBox(None)
        layout = box.layout()
        layout.addWidget(self.toolbar[name])
        layout.addWidget(self.canvas[name])
        self.tabs.addTab(box, name)

    def update_comment(self, comment, prefix=''):
        pass
        # Do nothing
        # super().update_comment(comment, prefix='Showing stats')

    def apply(self):
        try:
            comment = ''
            self.clear_messages()
            self.data_out = {}
            copy_items(self.data_in, self.data_out)
            module_data = super().get_data_by_module(self.data_in, self.module)
            if self.module in MODULE_MULTI:
                it = self.selScan.currentText()
                scan = module_data[it]
                scan_fit, comment = self.apply_scan(scan, scan_name=it, comment=comment)
            elif self.module in MODULE_SINGLE:
                scan_fit, comment = self.apply_scan(module_data)
            self.update_comment(comment)
            self.setStatusMessage(comment)
            self.info.set_output_summary(comment)
        except Exception as e:
            print(e)
            self.Error.unknown(repr(e))

    def apply_scan_item(self, Z, comment='', item=None):
        Zret = {}
        dims = super().get_detector_dimensions(Z)
        axes = dims.nonzero()[0][::-1]
        coef_x, Zerr, _ = fit_nD_metrology(Z, filter_terms_poly=[1, 1, 1, 1, 0, 1, 0], dtyp=item)

        if len(dims) < 3:  # non sequence
            coef_x = coef_x[np.newaxis, :]
        Z_avg_det = np.nanmean(Z, axis=tuple(axes), keepdims=True)
        Z_rms_det = np.sqrt(np.nanmean(np.square(Z - Z_avg_det), axis=tuple(axes)))
        tilt_y = 2 * coef_x[:, 1]
        tilt_x = 2 * coef_x[:, 2]
        curv_y = 2 * coef_x[:, 3]
        curv_x = 2 * coef_x[:, 5]
        unit_z = ''
        unit_tilt = ''
        unit_curv = ''
        try:
            if isinstance(Z, MetrologyData):
                uz = Z.unit
                unit_z = ' ({})'.format(uz)
                upix = Z.get_axis_val_items_detector()[-1].unit
                if item == 'height':
                    ut = (u.rad * uz / upix).decompose()
                else:
                    ut = uz
                unit_tilt = ' ({})'.format(ut)
                if item == 'height':
                    uc = (uz / (upix * upix)).decompose()
                else:
                    uc = (uz / upix).decompose()
                unit_curv = ' ({})'.format(uc)
        except Exception:
            pass

        self.figure[item].clear()
        ax = self.figure[item].add_subplot(311)
        ax.plot(tilt_x, '*-', label='Tilt_x - rms={:.2f}, pv={:.2f}'.format(rms(tilt_x), pv(tilt_x)))
        ax.plot(tilt_y, '*-', label='Tilt_y - rms={:.2f}, pv={:.2f}'.format(rms(tilt_y), pv(tilt_y)))
        ax.set(xlabel='Image number', ylabel='Tilt{}'.format(unit_tilt))
        ax.legend(loc='upper right')
        ax = self.figure[item].add_subplot(312)
        ax.plot(curv_x, '*-', label='Curvature_x - rms={:.2f}, pv={:.2f}'.format(rms(curv_x), pv(curv_x)))
        ax.plot(curv_y, '*-', label='Curvature_y - rms={:.2f}, pv={:.2f}'.format(rms(curv_y), pv(curv_y)))
        ax.set(xlabel='Image number', ylabel='Curvature{}'.format(unit_curv))
        ax.legend(loc='upper right')
        ax = self.figure[item].add_subplot(313)
        ax.plot(Z_rms_det.ravel(), '*-', label='rms={:.2f}, pv={:.2f}'.format(rms(Z_rms_det), pv(Z_rms_det)))
        ax.set(xlabel='Image number', ylabel='Rms of {} errors{}'.format(item, unit_z))
        ax.legend(loc='upper right')
        self.canvas[item].draw()
        return Zret, comment
