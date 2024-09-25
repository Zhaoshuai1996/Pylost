# coding=utf-8
import scipy.ndimage
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy, QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.config import config_params
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import copy_items
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWFilter(PylostWidgets, PylostBase):
    """Widget to filter, e.g. Gaussin on the datases"""
    name = 'Filter'
    description = 'Apply filters like low pass, high pass etc.'
    icon = "../icons/filter-sin.svg"
    priority = 52

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    source = Setting('', schema_only=True)
    sigma = Setting(3.0, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()

        box = super().init_info(module=True)
        self.btnApply = gui.button(box, self, 'Apply filter', callback=self.applyFilter, autoDefault=False, stretch=1,
                                   sizePolicy=QSizePolicy(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(self.controlArea, "Filter", stretch=1)
        items = ['None', 'Low_Pass_Gaussian', 'High_Pass_Gaussian']
        combo = gui.comboBox(box, self, "source", label='Filter:', sendSelectedValue=True, orientation=Qt.Horizontal,
                             labelWidth=50, stretch=1,
                             items=items, callback=self.change_source, sizePolicy=(Policy.Fixed, Policy.Fixed))
        combo.setSizePolicy(Policy.Fixed, Policy.Fixed)
        self.leSigma = gui.lineEdit(box, self, 'sigma', 'Sigma: ', orientation=Qt.Horizontal,
                                    sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.hide_filter_options()

    def sizeHint(self):
        return QSize(500, 50)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True)
        if data is None:
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        super().load_data()
        self.applyFilter()

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Applied filter')

    def hide_filter_options(self):
        """Hide filter options"""
        self.leSigma.parent().hide()

    def change_source(self):
        """Change source, low pass or high pass gaussain"""
        self.hide_filter_options()
        if self.source == 'Low_Pass_Gaussian':
            self.leSigma.parent().show()
        if self.source == 'High_Pass_Gaussian':
            self.leSigma.parent().show()

    def applyFilter(self):
        """Apply filter"""
        try:
            self.setStatusMessage('')
            self.clear_messages()
            self.info.set_output_summary('Filtering...')
            QtWidgets.qApp.processEvents()
            if self.source in ['None', '']:
                self.data_out = {}
                copy_items(self.data_in, self.data_out)
                self.info.set_output_summary('No filter applied')
                self.setStatusMessage('No filter applied')
                self.Outputs.data.send(self.data_out)
                if config_params.DEFAULT_CLOSE_WIDGETS_AFTER_APPLY:
                    self.close()
            else:
                super().apply_scans()
                self.Outputs.data.send(self.data_out)
        except Exception as e:
            print(e)
            self.Error.unknown(repr(e))

    def apply_scan_item_detector(self, Zdet, comment=''):
        """Apply filter for each detector image looped in the subaperture datasets, if more than one image exist"""
        Zdet_out = scipy.ndimage.gaussian_filter(Zdet, self.sigma)
        if isinstance(Zdet, MetrologyData):
            Zdet_out = Zdet_out * Zdet.unit
        comment = 'Applied {} filter with sigma {}'.format(self.source, self.sigma)
        if self.source == 'High_Pass_Gaussian':
            Zdet_out = Zdet - Zdet_out
        return Zdet_out, comment
