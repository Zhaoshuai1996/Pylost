# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.widget import Msg
from scipy.ndimage import rotate

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import copy_items
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets

DEG_TO_RAD = 0.0174533


class OWRotateData(PylostWidgets, PylostBase):
    name = 'Rotate Data'
    description = 'Rotate 2D images by given angle.'
    icon = "../icons/rotate.svg"
    priority = 44

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    angle = Setting(0.0, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)
        self.btnApply = gui.button(box, self, 'Rotate', callback=self.applyRotation, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(self.controlArea, "Options")
        gui.lineEdit(box, self, "angle", "Rotation angle (degrees)", orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.applyRotation)

    def sizeHint(self):
        return QSize(500, 50)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True)
        if data is None:
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        super().load_data()
        self.applyRotation()

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Applied rotation')

    def applyRotation(self):
        copy_items(self.data_in, self.data_out)
        if self.angle != 0:
            super().apply_scans()
        self.Outputs.data.send(self.data_out)

    def apply_scan_item(self, Z, comment='', item=None):
        Zret = Z
        dims = super().get_detector_dimensions(Z)
        axes = dims.nonzero()[0]
        if self.angle != 0:
            if len(axes) == 2:
                Zret = rotate(Zret, angle=self.angle, reshape=True, cval=np.nan, axes=axes, prefilter=False)
            # elif len(axes)==1: # TODO: Needs to be correctly implemented
            #     Zret = rotate(Zret.reshape(-1, 1) + np.array([0] * 2), angle=self.angle, reshape=True, cval=np.nan, axes=[1, 0]).T
            if isinstance(Z, MetrologyData):
                Zret = Z.copy_to(Zret)
            comment = 'rotated data by {:.3f} degrees'.format(self.angle)

        return Zret, comment
