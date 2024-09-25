# coding=utf-8
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import copy_items, heights_to_metrologydata, integrate_slopes
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWIntegrateSlopes(PylostWidgets, PylostBase):
    name = 'Integrate Slopes'
    description = 'Integrate slopes to heights.'
    icon = "../icons/integral.svg"
    priority = 71

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    interpolate = Setting(True)
    keep_slopes = Setting(False)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)
        self.btnApply = gui.button(box, self, 'Apply', callback=self.applyIntegral, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        gui.checkBox(self.controlArea, self, 'interpolate', 'Interpolate NaN values before integration')
        gui.checkBox(self.controlArea, self, 'keep_slopes', 'Keep slopes in the output')

    def sizeHint(self):
        return QSize(500, 50)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True)
        if data is None:
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        super().load_data()
        self.applyIntegral()

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Applied integration')

    def applyIntegral(self):
        self.clear_messages()
        copy_items(self.data_in, self.data_out)
        if not self.keep_slopes:
            if 'slopes_x' in self.data_out:
                del self.data_out['slopes_x']
            if 'slopes_y' in self.data_out:
                del self.data_out['slopes_y']
        super().apply_scans()
        self.Outputs.data.send(self.data_out)

    def apply_scan(self, scan, scan_name=None, comment=''):
        scan_fit = {}
        fit_comment = ''

        x = y = None
        copy_items(scan, scan_fit, deepcopy=True)
        try:
            pix_size = self.pix_size if hasattr(self, 'pix_size') else None
            for key in ['slopes_x']:
                if key in scan and isinstance(scan[key], MetrologyData):
                    if any(scan[key].dim_detector):
                        pix_size = scan[key].pix_size_detector
                    else:
                        axis_vals = scan[key].get_axis_val_items()
                        x = axis_vals[-1] if len(axis_vals) > 0 else None
                        y = axis_vals[-2] if len(axis_vals) > 1 else None
            if 'slopes_x' in scan:
                if 'slopes_y' in scan:
                    scan_fit['height'] = integrate_slopes(scan['slopes_x'], scan['slopes_y'], x=x, y=y, pix_sz=pix_size,
                                                          interpolate_nans=self.interpolate, method='frankot_chellappa')
                    if scan_fit['height'].shape[0] == 1:
                        scan_fit['height'] = scan_fit['height'].ravel()
                    scan_fit['height'] = heights_to_metrologydata(scan, scan_fit, pix_size, x=x, y=y)
                else:
                    scan_fit['height'] = integrate_slopes(scan['slopes_x'], x=x, y=y, pix_sz=pix_size, method='trapz')
                    scan_fit['height'] = heights_to_metrologydata(scan, scan_fit, pix_size, x=x, y=y)
            if not self.keep_slopes:
                if 'slopes_x' in scan_fit:
                    del scan_fit['slopes_x']
                if 'slopes_y' in scan_fit:
                    del scan_fit['slopes_y']
            fit_comment = 'Integrated slopes to heights'

        except Exception as e:
            self.Error.unknown(repr(e))

        return scan_fit, fit_comment
