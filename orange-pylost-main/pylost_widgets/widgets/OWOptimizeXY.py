# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy, QSizePolicy as Policy, QTableWidgetItem
from astropy.units import Quantity
from orangewidget.settings import Setting
from orangewidget.widget import Msg
from silx.gui.widgets.TableWidget import TableWidget

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import arr_to_pix, copy_items, get_dims
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWOptimizeXY(PylostWidgets, PylostBase):
    name = 'Optimize XY'
    description = 'Optimize subaperture X and Y positions, by shifting few pixels to get best correlation.'
    icon = "../icons/view.svg"
    priority = 73

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    upsample = Setting(100, schema_only=True)
    ovelap_ratio = Setting(0.9, schema_only=True)
    order = Setting(1, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)
        self.btnApply = gui.button(box, self, 'Optimize', callback=self.apply, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(self.controlArea, "Options", stretch=1)
        gui.lineEdit(box, self, 'upsample', 'Up sample factor', labelWidth=200, orientation=Qt.Horizontal,
                     sizePolicy=(QSizePolicy.Fixed, QSizePolicy.Fixed))
        gui.lineEdit(box, self, 'ovelap_ratio', 'Ovelap ratio', labelWidth=200, orientation=Qt.Horizontal,
                     sizePolicy=(QSizePolicy.Fixed, QSizePolicy.Fixed))
        gui.lineEdit(box, self, 'order', 'Order (k) \n[cross correlation between i and i+k]', labelWidth=200,
                     orientation=Qt.Horizontal,
                     sizePolicy=(QSizePolicy.Fixed, QSizePolicy.Fixed))

        box = gui.vBox(self.controlArea, "Results (offsets in pixels)", stretch=19)
        gui.label(box, self, 'Motor positions are offset by following.'
                             '\ne.g.motor_x (new) = motor_x (old) - x (init) + x (calc).'
                             '\nx (init) = offset from rounding of subpixel motor x shift'
                             '\nx (calc) = offset calculated which maximized cross correlation between i, i+k')
        self.tblOffsets = TableWidget(None)
        box.layout().addWidget(self.tblOffsets)

    def sizeHint(self):
        return QSize(500, 800)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True)
        if data is None:
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        super().load_data()
        # self.apply()

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Optimize position')

    def apply(self):
        copy_items(self.data_in, self.data_out)
        super().apply_scans()
        self.Outputs.data.send(self.data_out)

    def apply_scan(self, scan, scan_name=None, comment=''):
        scan_fit = {}
        copy_items(scan, scan_fit)
        for i, item in enumerate(self.DATA_NAMES):
            if item in scan:
                Z = scan[item]
                dims = get_dims(Z)

                scan_fit[item] = Z

                mx = np.array(scan['motor_X']).ravel() if 'motor_X' in scan else []
                my = np.array(scan['motor_Y']).ravel() if 'motor_Y' in scan else []
                pix_sz = scan['pix_size'] if 'pix_size' in scan else []
                if isinstance(Z, MetrologyData) and np.any(Z.motors):
                    pix_sz = Z.pix_size_detector
                    for m in Z.motors:
                        mx = Quantity(np.array(m['values']).ravel(), unit=m['unit']) if m['name'] == 'motor_X' else mx
                        my = Quantity(np.array(m['values']).ravel(), unit=m['unit']) if m['name'] == 'motor_Y' else my

                if len(mx) > 0:
                    Zf = Z.reshape((-1,) + tuple(np.array(Z.shape)[dims.astype(bool)]))
                    if len(my) == 0:
                        my = np.zeros_like(mx)

                    from skimage.registration import phase_cross_correlation
                    ox, xoff_i = arr_to_pix(mx, pix_sz[-1])
                    oy, yoff_i = arr_to_pix(my, pix_sz[-2])
                    # xoff_i = np.array(ox) - np.array((mx-mx[0])/pix_sz[-1])
                    # yoff_i = np.array(oy) - np.array((my-my[0])/pix_sz[-2])
                    xoff, yoff, oxn, oyn, omxn, omyn = (
                        [0] * len(mx), [0] * len(mx), [0] * len(mx), [0] * len(mx), [0] * len(mx), [0] * len(mx))
                    for i in np.arange(len(mx) - self.order):
                        j = i + self.order
                        dx = ox[j] - ox[i]
                        dy = oy[j] - oy[i]
                        si = (i, slice(dy, None) if dy >= 0 else slice(None, dy),
                              slice(dx, None) if dx >= 0 else slice(None, dx))
                        sj = (j, slice(None, -dy) if dy > 0 else slice(-dy, None),
                              slice(None, -dx) if dx > 0 else slice(-dx, None))
                        shifts, err, phasediff = phase_cross_correlation(Zf[si], Zf[sj],
                                                                         upsample_factor=self.upsample,
                                                                         overlap_ratio=self.ovelap_ratio)  # ,
                        # reference_mask=np.isfinite(Zf[si]),
                        # moving_mask=np.isfinite(Zf[sj]))
                        xoff[j] = shifts[0]
                        yoff[j] = shifts[1]
                        oxn[j] = ox[j] + xoff[j]
                        oyn[j] = oy[j] + yoff[j]
                    omxn, omyn = (np.min(mx) + oxn * pix_sz[-1], np.min(my) + oyn * pix_sz[-2])
                    if isinstance(Zf, MetrologyData):
                        Zf._copy_items()
                        Zf.update_motor('motor_X', omxn.value, omxn.unit)
                        Zf.update_motor('motor_Y', omyn.value, omyn.unit)
                    else:
                        scan_fit['motor_X'] = omxn
                        scan_fit['motor_Y'] = omyn
                    scan_fit[item] = Zf
                    self.fillTable(xoff_i, yoff_i, xoff, yoff)
                    comment = 'optimization offsets (pix) \nX: max={:.3f}, min={:.3f}\nY: max={:.3f}, min={:.3f}' \
                        .format(np.nanmax(xoff), np.nanmin(xoff), np.nanmax(yoff), np.nanmin(yoff))

        return scan_fit, comment

    def fillTable(self, xi, yi, xc, yc):
        arr = [xi, yi, xc, yc]
        cols = 4
        rows = len(xi)
        self.tblOffsets.setRowCount(rows)
        self.tblOffsets.setColumnCount(cols)
        self.tblOffsets.setHorizontalHeaderLabels("X (init);Y (init);X (calc);Y (calc)".split(";"))
        # header = self.tblOffsets.horizontalHeader()
        # for i in np.arange(cols):
        #     header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)#ResizeToContents)
        # header.setSectionResizeMode(cols-1, QtWidgets.QHeaderView.Stretch)

        for i in np.arange(rows):
            for j in np.arange(cols):
                val = QTableWidgetItem('{:.3f}'.format(arr[j][i]))
                val.setToolTip('{}'.format(arr[j][i]))
                val.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.tblOffsets.setItem(i, j, val)
