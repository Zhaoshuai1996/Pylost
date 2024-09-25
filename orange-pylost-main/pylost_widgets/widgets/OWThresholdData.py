# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import copy_items
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWThresholdData(PylostWidgets, PylostBase):
    name = 'Threshold Data'
    description = 'Rotate 2D images by given angle.'
    icon = "../icons/threshold.svg"
    priority = 51

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    dataset = Setting('', schema_only=True)
    max = Setting(0.0, schema_only=True)
    min = Setting(-0.0, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)
        self.btnApply = gui.button(box, self, 'Apply', callback=self.applyThreshold, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(self.controlArea, "Options")
        gui.comboBox(box, self, "dataset", label='Dataset', sendSelectedValue=True, orientation=Qt.Horizontal,
                     labelWidth=50, stretch=1, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, "min", "Min value", orientation=Qt.Horizontal, sizePolicy=(Policy.Fixed, Policy.Fixed),
                     callback=self.applyThreshold)
        gui.lineEdit(box, self, "max", "Max value", orientation=Qt.Horizontal, sizePolicy=(Policy.Fixed, Policy.Fixed),
                     callback=self.applyThreshold)

    def sizeHint(self):
        return QSize(300, 50)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True)
        if data is None:
            self.controls.dataset.clear()
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        super().load_data()
        self.controls.dataset.clear()
        if len(self.DATA_NAMES) > 0:
            self.controls.dataset.addItems(self.DATA_NAMES)
            self.dataset = self.DATA_NAMES[0]
        self.applyThreshold()

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Applied threshold')

    def applyThreshold(self):
        copy_items(self.data_in, self.data_out)
        if self.min != 0 or self.max != 0:
            super().apply_scans()
        self.Outputs.data.send(self.data_out)

    def apply_scan(self, scan, scan_name=None, comment=''):
        scan_result = {}
        copy_items(scan, scan_result)
        if self.dataset in scan:
            Z = np.empty_like(scan[self.dataset])
            np.copyto(Z, scan[self.dataset])
            Zv = Z.value if isinstance(Z, MetrologyData) else Z
            if self.max != 0:
                Z[Zv > self.max] = np.nan
            if self.min != 0:
                Z[Zv < self.min] = np.nan
            scan_result[self.dataset] = Z
            comment = 'min {:.3f}, max {:.3f}'.format(self.min, self.max)

        return scan_result, comment
