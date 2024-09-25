# coding=utf-8
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.utils.signals import MultiInput
from orangewidget.widget import Msg

from pylost_widgets.util.util_functions import copy_items
from pylost_widgets.widgets._Operator import Operator
from pylost_widgets.widgets._PylostBase import PylostWidgets


class OWOperatorMultiply(PylostWidgets, Operator):
    name = 'Multiply'
    description = 'Multiply two sets of data.'
    icon = "../icons/multiply.svg"
    priority = 81

    class Inputs:
        data = MultiInput('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    pad_align = Setting(Operator.NONE, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        Operator.__init__(self)

        box = super().init_info(module=True)
        self.selModule.parent().hide()

        gui.comboBox(self.controlArea, self, 'pad_align', label='Align data at the selection and pad with NaN',
                     orientation=Qt.Horizontal,
                     items=self.ALIGN_OPT, callback=self.load_module, sizePolicy=(Policy.Fixed, Policy.Fixed))

    def sizeHint(self):
        return QSize(500, 50)

    @Inputs.data
    def set_data(self, index, data):
        self.data_index[index] = data
        super().init_data()

    @Inputs.data.insert
    def insert_data(self, index, data):
        self.data_index.insert(index, data)
        super().init_data()

    @Inputs.data.remove
    def remove_data(self, index):
        self.data_index.pop(index)
        super().init_data()

    def apply_scan(self, scan_result, scan=None, comment=''):
        if scan is None or not any(scan):
            return scan_result
        if not any(scan_result):
            copy_items(scan, scan_result)
            return scan_result

        for item in self.DATA_NAMES:
            if item in scan and item in scan_result:
                Zr, Z = super().pad_items(scan_result[item], scan[item], align_type=self.pad_align)
                scan_result[item] = Zr * Z
        return scan_result
