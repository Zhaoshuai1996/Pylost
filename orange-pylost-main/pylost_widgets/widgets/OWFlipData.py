# coding=utf-8
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.util.util_functions import flip_data
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWFlipData(PylostWidgets, PylostBase):
    name = 'Flip Data'
    description = 'Flip data i.e. rotate by 180 degrees. By default flipped in all detector dimensions, e.g. X and Y.'
    icon = "../icons/flip.svg"
    priority = 43

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    flipX = Setting(False, schema_only=True)
    flipY = Setting(False, schema_only=True)
    scale = Setting(1.0, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)
        self.btnApply = gui.button(box, self, 'Flip data', callback=self.applyFlip, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(self.controlArea, "Select dimensions to flip")
        gui.checkBox(box, self, "flipX", "Flip X", callback=self.change_fx)
        gui.checkBox(box, self, "flipY", "Flip Y")
        gui.lineEdit(box, self, 'scale', 'Scale data', orientation=Qt.Horizontal)

    def sizeHint(self):
        return QSize(500, 50)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True)
        if data is None:
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        super().load_data()
        self.applyFlip()

    def change_fx(self):
        pass

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Applied flip')

    def applyFlip(self):
        try:
            super().apply_scans()
            self.Outputs.data.send(self.data_out)
        except Exception as e:
            self.Error.unknown(str(e))

    def apply_scan_item(self, Z, comment='', item=None):
        Zret = self.scale * Z
        dims = super().get_detector_dimensions(Z)
        axes = dims.nonzero()[0][::-1]
        txt = ''
        if self.flipX and len(axes) > 0:
            Zret = flip_data(Zret, axes[0], flip_motors=['x'])  # np.flip(scan_fit[item], axis=axes[0])
            txt += 'X'
        if self.flipY and len(axes) > 1:
            Zret = flip_data(Zret, axes[1], flip_motors=['y'])  # np.flip(scan_fit[item], axis=axes[1])
            txt += 'Y'
        comment = 'Flipped data along {} axis. '.format(txt) if txt != '' else 'No axes flipped. '
        if self.scale != 1:
            comment += 'Applied scaling factor {}'.format(self.scale)

        return Zret, comment
