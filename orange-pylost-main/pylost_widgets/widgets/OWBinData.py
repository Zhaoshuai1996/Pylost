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


class OWBinData(PylostWidgets, PylostBase):
    """Bin data along X,Y directions"""
    name = 'Bin Data'
    description = 'Bin data'
    icon = "../icons/reduce.svg"
    priority = 54

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    binX = Setting(1, schema_only=True)
    binY = Setting(1, schema_only=True)
    MEDIAN, MEAN, FIRST, LAST = range(4)
    options = ['Median', 'Mean', 'First pixel', 'Last pixel']
    reduction = Setting(MEDIAN, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)
        self.btnApply = gui.button(box, self, 'Apply', callback=self.applyBin, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(self.controlArea, "Select dimensions to flip")
        gui.lineEdit(box, self, "binX", "Bin size X (pixels)", orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, "binY", "Bin size Y (pixels)", orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.comboBox(box, self, 'reduction', label='Binning method', items=self.options, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))

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
        self.applyBin()

    def update_comment(self, comment, prefix=''):
        """Implementation in super class PylostBase is used"""
        super().update_comment(comment, prefix='Applied bin data')

    def applyBin(self):
        """Callback from applying binning. Implementation in super class PylostBase is used"""
        super().apply_scans()
        self.Outputs.data.send(self.data_out)

    def apply_scan(self, scan, scan_name=None, comment=''):
        """
        Apply binning for each scan.

        :param scan: Scan dictionary with height/slope information
        :type scan: dict
        :param scan_name: Name of the scan, useful if many repeated scans are present
        :type scan_name: str
        :param comment: Comment text
        :type comment: str
        :return: (Result dictionary with processed scan data, comment)
        :rtype: (dict, str)
        """
        scan_fit = {}
        fit_comment = ''
        copy_items(scan, scan_fit)
        if self.binX > 1 or self.binY > 1:
            for i, item in enumerate(self.DATA_NAMES):
                if item in scan:
                    Z = scan[item]
                    dims = self.get_detector_dimensions(Z)
                    axes = dims.nonzero()[0][::-1]
                    if isinstance(Z, MetrologyData):
                        self.pix_size = scan[item].pix_size

                    scan_fit[item] = Z.copy()
                    # scan_fit[item]._copy_items()
                    txt = ''
                    pix_sz_new = self.pix_size.copy()
                    if self.binX > 1 and len(axes) > 0:
                        scan_fit[item] = self.get_binned_data(scan_fit[item], axes[0], self.binX)
                        pix_sz_new[axes[0]] = self.pix_size[axes[0]] * self.binX
                        txt = '{} X axis with {} pix;'.format(txt, self.binX)
                    if self.binY > 1 and len(axes) > 1:
                        scan_fit[item] = self.get_binned_data(scan_fit[item], axes[1], self.binY)
                        pix_sz_new[axes[1]] = self.pix_size[axes[1]] * self.binY
                        txt = '{} Y axis with {} pix;'.format(txt, self.binY)
                    if isinstance(Z, MetrologyData):
                        scan_fit[item] = Z.copy_to(scan_fit[item])
                        scan_fit[item]._set_pix_size(pix_sz_new)
                    fit_comment = 'Binned data along {}'.format(txt) if txt != '' else 'No binning'

        return scan_fit, fit_comment

    def get_binned_data(self, data, axis, bin_sz):
        """
        Get binned dataset

        :param data: Dataset to bin
        :type data: MetrologyData/numpy.ndarray
        :param axis: Binning axis
        :type axis: int
        :param bin_sz: Bin size
        :type bin_sz: int
        :return: Binned dataset
        :rtype: MetrologyData/numpy.ndarray
        """
        bin_sz = int(bin_sz)
        n_bins = data.shape[axis] // bin_sz
        data = data.take(indices=np.arange(n_bins * bin_sz), axis=axis)
        shp = data.shape
        if self.reduction == self.FIRST:
            ind = np.arange(0, data.shape[axis], bin_sz)
            return data.take(indices=ind, axis=axis)
        elif self.reduction == self.LAST:
            ind = np.arange(bin_sz - 1, data.shape[axis], bin_sz)
            return data.take(indices=ind, axis=axis)
        else:
            new_shp = list(shp)
            new_shp[axis] = bin_sz
            new_shp.insert(axis, n_bins)
            new_data = data.reshape(new_shp)
            if self.reduction == self.MEDIAN:
                return np.nanmedian(new_data, axis + 1)
            elif self.reduction == self.MEAN:
                return np.nanmean(new_data, axis + 1)
        return None
