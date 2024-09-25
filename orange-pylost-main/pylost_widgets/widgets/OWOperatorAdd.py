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


class OWOperatorAdd(PylostWidgets, Operator):
    name = 'Add / Average'
    description = 'Add or average data sets.'
    icon = "../icons/plus.svg"
    priority = 41

    class Inputs:
        data = MultiInput('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    average = Setting(False, schema_only=True)
    pad_align = Setting(Operator.NONE, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        Operator.__init__(self)

        self.scale = 1

        box = super().init_info(module=True)
        self.selModule.parent().hide()
        gui.checkBox(self.controlArea, self, 'average', 'Average (divide by number of inputs)',
                     callback=self.change_average)

        gui.comboBox(self.controlArea, self, 'pad_align', label='Align data at the selection and pad with NaN',
                     orientation=Qt.Horizontal,
                     items=self.ALIGN_OPT, callback=self.load_module, sizePolicy=(Policy.Fixed, Policy.Fixed))

    def sizeHint(self):
        return QSize(500, 50)

    @Inputs.data
    def set_data(self, index, data):
        self.data_index[index] = data
        self._highest_in_front()
        super().init_data()
        self.change_average()

    @Inputs.data.insert
    def insert_data(self, index, data):
        self.data_index.insert(index, data)
        self._highest_in_front()
        super().init_data()
        self.change_average()

    @Inputs.data.remove
    def remove_data(self, index):
        self.data_index.pop(index)
        super().init_data()
        self.change_average()

    def change_average(self):
        if len(self.data_in) > 0:
            self.scale = len(self.data_in) - 1 if self.average else 1
            self.load_module()

    def _highest_in_front(self):
        '''put the highest number of subapertures in first position to keep the data structures'''
        if len(self.data_in) < 3:  # include log
            return
        argmax = 0
        for item in self.DATA_NAMES:
            for idx, data in enumerate(self.data_index):
                if data is None:
                    continue
                sub_dim = 0 if len(data[item].shape_non_detector) < 1 else data[item].shape_non_detector[0]
                argmax = idx if sub_dim > argmax else argmax
        if argmax > 0:
            self.data_index.insert(0, self.data_index.pop(argmax))

    def apply_scan(self, scan_result, scan=None, comment=''):
        if scan is None or not any(scan):
            cmt = 'no data'
            self.update_comment(cmt)
            self.setStatusMessage(cmt)
            return scan_result
        if not any(scan_result):
            copy_items(scan, scan_result, deepcopy=True)
            for item in self.DATA_NAMES:
                if item in scan and item in scan_result:
                    scan_result[item] = scan_result[item] / self.scale
            cmt = 'data added'
            if self.average:
                cmt = 'data averaged'
            self.update_comment(cmt)
            self.setStatusMessage(cmt)
            return scan_result

        for item in self.DATA_NAMES:
            if item in scan and item in scan_result:
                Zr, Z = super().pad_items(scan_result[item], scan[item], align_type=self.pad_align)
                scan_result[item] = Zr + (Z / self.scale)
        cmt = 'data added'
        if self.average:
            cmt = 'data averaged'
        self.update_comment(cmt)
        self.setStatusMessage(cmt)
        return scan_result
