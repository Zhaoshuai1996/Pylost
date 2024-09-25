# coding=utf-8

import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.utils.signals import MultiInput
from orangewidget.widget import Msg

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.widgets._Operator import Operator
from pylost_widgets.widgets._PylostBase import PylostWidgets


class OWOperatorMerge(PylostWidgets, Operator):
    name = 'Merge'
    description = 'Merge data sets.'
    icon = "../icons/merge.svg"
    priority = 62

    class Inputs:
        data = MultiInput('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    pad_align = Setting(Operator.NONE, schema_only=True)
    NEW, FIRST, LAST, SELECT = np.arange(4)
    axis = Setting(FIRST)
    sel_axis = Setting(0, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        Operator.__init__(self)

        box = super().init_info(module=True)
        self.selModule.parent().hide()
        gui.radioButtons(self.controlArea, self, 'axis',
                         ['New axis', 'First axis', 'Last axis (X)', 'Select another axis'], label='Select axis',
                         callback=self.change_radio)
        gui.lineEdit(self.controlArea, self, 'sel_axis', 'Select merge axis', sizePolicy=(Policy.Fixed, Policy.Fixed))
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

    def change_radio(self):
        if self.axis == self.FIRST:
            self.sel_axis = 0
        if self.axis == self.LAST:
            self.sel_axis = -1
        self.load_module()

    def apply_scan(self, scan_result, scan=None, comment=''):
        if scan is None or not any(scan):
            return scan_result

        for item in scan:
            if item not in scan_result:
                if isinstance(scan[item], MetrologyData):
                    scan_result[item] = scan[item].view(np.ndarray)
                    scan_result[item] = scan[item].copy_to(scan_result[item])
                else:
                    scan_result[item] = scan[item]
            else:
                if item in self.DATA_NAMES:
                    Zr, Z = super().pad_items(scan_result[item], scan[item], align_type=self.pad_align)
                else:
                    Zr, Z = scan_result[item], scan[item]
                if isinstance(Z, np.ndarray):
                    axis = self.sel_axis
                    motors_res = []
                    if isinstance(Zr, MetrologyData):
                        motors_res = Zr.motors
                    if self.axis == self.NEW:
                        if Zr.ndim == Z.ndim:
                            scan_result[item] = np.stack([Zr, Z], axis=0)
                        elif Zr.ndim == Z.ndim + 1:
                            scan_result[item] = np.concatenate([Zr, Z[np.newaxis]], axis=0)
                        else:
                            raise Exception('Shape mismatch')
                    else:
                        scan_result[item] = np.concatenate([Zr, Z], axis=axis)
                    try:
                        if isinstance(Z, MetrologyData):
                            motors = Z.motors
                            for i, m in enumerate(motors):
                                motors_res[i]['values'] = np.append(motors_res[i]['values'], motors[i]['values'])
                            scan_result[item].update_motors(motors_res)
                    except Exception as e:
                        print('Error updating motors after merge')
                elif isinstance(Z, list):
                    scan_result[item] += Z
                else:
                    scan_result[item] = Z

        return scan_result
