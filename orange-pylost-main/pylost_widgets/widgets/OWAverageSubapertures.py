# coding=utf-8
import numpy as np
from Orange.widgets import widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWAverageSubapertures(PylostWidgets, PylostBase):
    """Average data along subaperture dimension (or all the non detector dimensions if ndim>3)"""
    name = 'Average Subapertures'
    description = 'Average subapertures data.'
    icon = "../icons/avg.svg"
    priority = 63

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)

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
        self.calc_average()

    def update_comment(self, comment='', prefix=''):
        """Implementation in super class PylostBase is used"""
        super().update_comment(comment, prefix='Applied average subapertures')

    def calc_average(self):
        """Callback from applying average subapertures. Implementation in super class PylostBase is used"""
        super().apply_scans()
        self.Outputs.data.send(self.data_out)

    def apply_scan_item(self, Z, comment='', item=None):
        """
        Average subapertures for each scan item

        :param Z: Scan item
        :type Z: MetrologyData
        :param comment: Comment text
        :type comment: str
        :param item: Item name, e.g. height
        :type item: str
        :return: Processed dataset, comment text
        :rtype: (MetrologyData/numpy.ndarray, str)
        """
        dims = self.get_detector_dimensions(Z)
        if any(dims) and False in dims:
            dims_nd = np.invert(dims)
            axes_nd = dims_nd.nonzero()[0]
            shp_nd = np.asarray(Z.shape)[dims_nd]
            return np.nanmean(Z, axis=tuple(axes_nd)), 'averaged {} subapertures'.format(np.sum(shp_nd))
        else:
            return Z, 'no subapertures found'
