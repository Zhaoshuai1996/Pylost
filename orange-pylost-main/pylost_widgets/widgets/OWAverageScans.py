# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.config import config_params
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import MODULE_MULTI, copy_items, parseQInt
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWAverageScans(PylostWidgets, PylostBase):
    """Widget to average repeated scans"""
    name = 'Average scans'
    description = 'Average raw/stitched scan data.'
    icon = "../icons/avg-scans.svg"
    priority = 64

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    scan_query = Setting('', schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)
        self.qscans = []
        self.qscans_str = []

        box = super().init_info(module=True, module_callback=self.change_module)
        gui.lineEdit(self.controlArea, self, 'scan_query', 'Scans to average: ', labelWidth=150,
                     orientation=Qt.Horizontal, sizePolicy=(Policy.Fixed, Policy.Fixed),
                     callback=self.change_scan_query)

    def sizeHint(self):
        return QSize(500, 50)

    @Inputs.data
    def set_data(self, data):
        """
        Linked to data input channel. Implementation in super class PylostBase is used

        :param data: Input data
        :type data: dict
        """
        super().set_data(data, update_names=True)
        if data is None:
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        """Implementation in super class PylostBase is used"""
        super().load_data()
        self.change_scan_query()

    def update_comment(self, comment='', prefix=''):
        """Implementation in super class PylostBase is used"""
        super().update_comment(comment, prefix='Applied average scans')

    def change_module(self):
        """Re-implemented as the output module can change. E.g. if input module is scan_data, after averaging scans, module changes to custom"""
        # Just output average of scans
        self.clear_messages()
        copy_items(self.data_in, self.data_out, deepcopy=True)
        if self.module in MODULE_MULTI:
            scans = self.get_data_by_module(self.data_in, self.module)
            scan_avg, count = self.average_scans(scans)

            # if self.module == 'stitch_data':
            #     save_module = 'stitch_avg'
            if self.module == 'scan_data':
                save_module = 'custom'
                del self.data_out['scan_data']
            self.set_data_by_module(self.data_out, save_module, scan_avg)
            cmt = "Average of {} scans".format(count)
        else:
            cmt = "No scans to average"
            self.Error.unknown('Scans not found')
        self.update_comment(cmt)
        self.setStatusMessage(cmt)
        self.info.set_output_summary(cmt)
        self.Outputs.data.send(self.data_out)

    def change_scan_query(self):
        """Callback linked to selection query, to average only selected scans"""
        self.qscans = parseQInt(self.scan_query)
        self.qscans_str = ['Scan_{}'.format(x) for x in self.qscans]
        self.change_module()

    def average_scans(self, scans):
        """
        Function with main logic to average scans

        :param scans: Repeated scans
        :type scans: dict
        :return: Averaged scan, number of averaged scans
        :rtype: dict, int
        """
        try:
            scan_avg = {}
            item_dict = {}
            for item in self.DATA_NAMES:
                item_dict[item] = []
                for key in scans:
                    if len(self.qscans_str) > 0 and key not in self.qscans_str:
                        continue
                    scan = scans[key]
                    if item in scan:
                        item_dict[item] += [scan[item]]
            scan_0 = list(scans.values())[0]
            cnt = 0
            for item in self.DATA_NAMES:
                cnt = np.max([cnt, len(item_dict[item])])
                if len(item_dict[item]) > 0:
                    scan_avg[item] = np.nanmean(np.array(item_dict[item]), axis=0)
                    if isinstance(scan_0[item], MetrologyData):
                        scan_avg[item] = scan_0[item].copy_to(scan_avg[item])
            if config_params.DEFAULT_CLOSE_WIDGETS_AFTER_APPLY:
                self.close()
            return scan_avg, cnt
        except Exception as e:
            return self.Error.unknown(repr(e)), 0
