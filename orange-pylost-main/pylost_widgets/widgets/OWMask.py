# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5 import QtCore
from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QGridLayout, QSizePolicy as Policy, QTabWidget
from orangewidget.settings import Setting
from orangewidget.widget import Msg
from silx.gui.plot import Plot1D

from pylost_widgets.config import config_params
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import MODULE_MULTI, MODULE_SINGLE, apply_mask, copy_items, get_pix_size
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWMask(PylostWidgets, PylostBase):
    name = 'Mask'
    description = 'Apply stitch mask to raw data'
    icon = "../icons/mask.svg"
    priority = 21

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 0

    module = Setting('', schema_only=True)
    scan_name = Setting('')
    # mask = Setting(bytes(), schema_only=True)
    # mask_shape = Setting(tuple(), schema_only=True)
    DEFAULT, SUBAP, SUBAP_SCANS, SCAN = range(4)
    option = Setting(DEFAULT, schema_only=True)

    view_pixels = Setting(False, schema_only=True)
    relative = Setting(True, schema_only=True)
    cx = Setting(0.0, schema_only=True)
    cy = Setting(0.0, schema_only=True)
    w = Setting(0.0, schema_only=True)
    h = Setting(0.0, schema_only=True)
    sigMaskParams = pyqtSignal(float, float, float, float, bool, bool)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)
        self.move_rect = False
        self.mask_tools = None

        box = super().init_info(module=True, module_callback=self.change_module, scans=True,
                                scans_callback=self.change_scan)
        self.btnApply = gui.button(box, self, 'Apply mask', callback=self.applyUserMask, stretch=1, autoDefault=False,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        hbox = gui.hBox(self.controlArea)
        box = gui.vBox(hbox, 'Mask options')
        gui.checkBox(box, self, 'view_pixels', 'View Pixels', callback=[self.check_pixel_view, self.change_mask_params])
        gui.checkBox(box, self, 'relative', 'Relative to data center', callback=self.change_mask_params)
        self.lblCenterCam = gui.label(box, self, '')
        self.lblCenterData = gui.label(box, self, '')

        box = gui.vBox(hbox, 'Mask draw')
        layout = QGridLayout()
        gui.widgetBox(box, orientation=layout)
        self.lblX = gui.label(None, self, 'X')
        self.lblY = gui.label(None, self, 'Y')
        layout.addWidget(self.lblX, 0, 1, alignment=Qt.AlignCenter)
        layout.addWidget(self.lblY, 0, 2, alignment=Qt.AlignCenter)
        lbl = gui.label(None, self, 'Center')
        le1 = gui.lineEdit(None, self, 'cx', None, controlWidth=50, alignment=Qt.AlignRight,
                           callback=self.change_mask_params)
        le2 = gui.lineEdit(None, self, 'cy', None, controlWidth=50, alignment=Qt.AlignRight,
                           callback=self.change_mask_params)
        layout.addWidget(lbl, 1, 0)
        layout.addWidget(le1, 1, 1)
        layout.addWidget(le2, 1, 2)
        lbl = gui.label(None, self, 'Size')
        le1 = gui.lineEdit(None, self, 'w', None, controlWidth=50, alignment=Qt.AlignRight,
                           callback=self.change_mask_params)
        le2 = gui.lineEdit(None, self, 'h', None, controlWidth=50, alignment=Qt.AlignRight,
                           callback=self.change_mask_params)
        layout.addWidget(lbl, 2, 0)
        layout.addWidget(le1, 2, 1)
        layout.addWidget(le2, 2, 2)
        btn = gui.button(box, self, 'Draw', callback=self.draw_mask, autoDefault=False,
                         sizePolicy=(Policy.Fixed, Policy.Fixed))
        box.layout().addWidget(btn, alignment=Qt.AlignRight)

        box = gui.vBox(hbox, 'Mask tools')
        gui.button(box, self, 'Save', callback=self.save_mask, autoDefault=False,
                   sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.button(box, self, 'Load', callback=self.load_mask, autoDefault=False,
                   sizePolicy=(Policy.Fixed, Policy.Fixed))
        button = gui.button(box, self, 'Move', callback=self.move_mask, autoDefault=False,
                            sizePolicy=(Policy.Fixed, Policy.Fixed))
        button.setFocusPolicy(QtCore.Qt.NoFocus)

        box = gui.vBox(hbox, 'Options', stretch=1)
        rbox = gui.radioButtons(box, self, "option", box=False, addSpace=True)
        gui.appendRadioButton(rbox, 'Default')
        gui.appendRadioButton(rbox, 'Apply mask only for current subaperture')
        self.rb3 = gui.appendRadioButton(rbox, 'Apply mask for current subaperture in all scans')
        self.rb4 = gui.appendRadioButton(rbox, 'Apply mask for all subapertures in the current scan')
        gui.label(box, self, '')

        # Data viewer
        box1 = gui.vBox(self.controlArea, "Data viewer", stretch=19)
        self.tabs = QTabWidget(self)
        self.dataViewers = {}
        box1.layout().addWidget(self.tabs)

    def sizeHint(self):
        return QSize(500, 500)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_tabs=True)
        if data is None:
            self.Outputs.data.send(None)
            self.no_input()

    def activateWindow(self):
        self.draw_mask()
        super().activateWindow()

    def show(self):
        self.draw_mask()
        super().show()

    def focusInEvent(self, event):
        self.draw_mask()
        super().focusInEvent(event)

    def change_mask_params(self):
        self.sigMaskParams.emit(self.cx, self.cy, self.w, self.h, self.view_pixels, self.relative)

    def set_mask_inputs(self, cx, cy, w, h):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

    def check_pixel_view(self, update_inputs=True):
        self.blockSignals(True)
        dv = self.dataViewers[self.DATA_NAMES[self.tabs.currentIndex()]]
        dims = self.get_detector_dimensions(dv.data())
        axes = ['X', 'Y']
        units = ['pix', 'pix']
        shape = dv.data().shape
        axis_vals = [[x for x in range(shape[-1])], [y for y in range(shape[-2])]]
        if isinstance(dv.data(), MetrologyData):
            axis_vals = dv.get_selection(dv.data().get_axis_val_items())[-2:]
            axes = dv.get_selection(dv.data().axis_names)
            if sum(dims) < 2:
                axes = ['-'] + axes
            if sum(dims) >= 2:
                units = ['{}'.format(axis_vals[-1].unit), '{}'.format(axis_vals[-2].unit)]
            else:
                units = ['{}'.format(axis_vals[-1].unit)] * 2
        if self.view_pixels:
            self.lblX.setText('{} (pix)'.format(axes[-1]))
            self.lblY.setText('{} (pix)'.format(axes[-2]))
        else:
            self.lblX.setText('{} ({})'.format(axes[-1], units[-1]))
            self.lblY.setText('{} ({})'.format(axes[-2], units[-2]))
        if update_inputs and sum(dims) >= 2:
            mt = dv.get_mask_tools()
            sx, sy = [1, 1] if mt is None else mt._scale  # pixel sizes
            if self.view_pixels:  # length to pixel
                center = dv.data().center if self.relative else dv.data().center_absolute
                center_pix = dv.data().center_pix if self.relative else dv.data().center_absolute_pix
                cx = self.cx + int(round(center[1].value / sx)) - center_pix[1]
                cy = self.cy + int(round(center[0].value / sy)) - center_pix[0]
                w = self.cx + int(round(self.w / sx))
                h = self.cy + int(round(self.h / sy))
                self.set_mask_inputs(int(cx), int(cy), int(w), int(h))
            else:  # pixel to length
                self.set_mask_inputs(np.round(self.cx * sx, 3), np.round(self.cy * sy, 3),
                                     np.round(self.w * sx, 3), np.round(self.h * sy, 3))
        self.blockSignals(False)

    def save_mask(self):
        pass

    def load_mask(self):
        pass

    def move_mask(self):
        """
        Enable/disable move mask option.
        :return:
        """
        if self.move_rect:
            self.disable_move_rect()
        else:
            QApplication.setOverrideCursor(Qt.SizeAllCursor)
            self.move_rect = True

    def disable_move_rect(self):
        """
        Disable move mask option.
        :return:
        """
        if self.move_rect:
            QApplication.restoreOverrideCursor()
            self.move_rect = False

    def move_center_px(self, dx, dy):
        """
        Move center by given pixels
        :param dx: pixels along x
        :param dy: pixels along y
        :return:
        """
        try:
            if dx or dy:
                dv = self.dataViewers[self.DATA_NAMES[self.tabs.currentIndex()]]
                mt = dv.get_mask_tools()
                sx, sy = [1, 1] if self.view_pixels else mt._scale
                self.cx = np.round(self.cx + dx * sx, 5)
                self.cy = np.round(self.cy + dy * sy, 5)
                if self.w > 0 and self.h > 0:
                    for key in self.dataViewers:
                        self.dataViewers[key].drawRectMask(cx=self.cx, cy=self.cy, sx=self.w, sy=self.h,
                                                           check_pixels=self.view_pixels, check_relative=self.relative)
                self.change_mask_params()
        except Exception as e:
            pass

    def mousePressEvent(self, event):
        """
        Disable move mask option (if it is on), with mouse click anywhere.
        :param event:
        :return:
        """
        self.disable_move_rect()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        """
        Move mask based on key presses for numbers 2,4,6,8
        :param event:
        :return:
        """
        if self.move_rect:
            if event.key() == QtCore.Qt.Key_Q:
                self.disable_move_rect()
            elif event.key() == QtCore.Qt.Key_Left or event.key() == QtCore.Qt.Key_4:
                self.move_center_px(-1, 0)
            elif event.key() == QtCore.Qt.Key_Right or event.key() == QtCore.Qt.Key_6:
                self.move_center_px(1, 0)
            elif event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_8:
                self.move_center_px(0, 1)
            elif event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_2:
                self.move_center_px(0, -1)
        super().keyPressEvent(event)

    def draw_mask(self):
        for key in self.dataViewers:
            dv = self.dataViewers[key]
            dims = self.get_detector_dimensions(dv.data())
            if sum(dims) < 2:
                dv.drawROI(cx=self.cx, w=self.w, check_relative=self.relative)
            else:
                dv.drawRectMask(cx=self.cx, cy=self.cy, w=self.w, h=self.h, check_pixels=self.view_pixels,
                                check_relative=self.relative)

    def load_data(self, multi=False):
        super().load_data()
        self.change_module()
        for key in self.dataViewers:
            self.dataViewers[key].sigMaskParams.connect(self.update_mask_params)
        self.Outputs.data.send(self.data_out)

    def update_mask_params(self, cx, cy, w, h, view_pixels, relative):
        self.blockSignals(True)
        self.set_mask_inputs(cx, cy, w, h)
        self.view_pixels = view_pixels
        self.relative = relative
        self.blockSignals(False)

    def change_module(self):
        self.selScan.setEnabled(False)
        self.selScan.clear()
        self.clear_viewers()
        module_data = self.get_data_by_module(self.data_in, self.module)
        if self.module in MODULE_MULTI:
            self.rb3.show()
            self.rb4.show()
            if len(module_data) > 0:
                self.selScan.setEnabled(True)
                self.selScan.addItems(list(module_data.keys()))
                self.change_scan()
        elif self.module in MODULE_SINGLE:
            self.rb3.hide()
            self.rb4.hide()
            self.load_viewer(module_data, show_mask=True)
            self.applyDefMask()

    def change_scan(self):
        curScan = self.selScan.currentText()
        scans = self.get_data_by_module(self.data_in, self.module)
        self.load_viewer(scans[curScan], show_mask=True)
        self.applyDefMask()

    def no_input(self):
        self.selScan.clear()
        self.clear_viewers()
        self.infoInput.setText("No data on input yet, waiting to get something.")

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Applied mask')

    def applyDefMask(self):
        # Befroe applying default mask apply parameters and update dataviewer mask parameters
        self.check_pixel_view(update_inputs=False)
        self.change_mask_params()
        self.draw_mask()
        self.applyUserMask()

    def applyUserMask(self):
        dv = self.dataViewers[self.DATA_NAMES[self.tabs.currentIndex()]]
        mask = dv.getMask()
        try:
            if mask is None and dv.findChild(Plot1D) is not None:
                plt = dv.findChild(Plot1D)
                if plt is not None:
                    roi = plt.getCurvesRoiDockWidget().currentROI
                    if roi is not None:
                        x = plt.getActiveCurve().getXData(copy=False)
                        mask = np.multiply(x > roi.getFrom(), x < roi.getTo())
                        self.cx = np.round((roi.getFrom() + roi.getTo()) / 2, 3)
                        self.w = np.round(roi.getTo() - roi.getFrom(), 3)
                        self.cy = 0
                        self.h = 0
        except Exception as e:
            self.Error.unknown(str(e))
        self.applyMask(mask)

    def applyMask(self, mask):
        self.clear_messages()
        if np.any(mask):
            pix_sz = None
            result = {}
            dv = self.dataViewers[self.DATA_NAMES[self.tabs.currentIndex()]]
            self.selection, _ = dv.get_selection_axes()
            module_data = self.get_data_by_module(self.data_in, self.module)
            if self.module in MODULE_MULTI:
                for it in module_data:
                    scan = module_data[it]
                    if self.option in [self.SCAN, self.SUBAP]:
                        if it == self.selScan.currentText():
                            scan_mask, pix_sz, ok = self.maskScan(scan, mask)
                        else:
                            ok = False
                    else:
                        scan_mask, pix_sz, ok = self.maskScan(scan, mask)
                    result[it] = scan_mask if ok else scan
            elif self.module in MODULE_SINGLE:
                result, pix_sz, ok = self.maskScan(module_data, mask)
                if not ok:
                    result = module_data
            self.set_data_by_module(self.data_out, self.module, result)
            self.Outputs.data.send(self.data_out)
            if (not self.view_pixels) and pix_sz is not None and len(pix_sz) > 0:
                if len(pix_sz) == 1:
                    w = self.w * pix_sz[0].unit
                    cx = self.cx * pix_sz[0].unit
                    cmt = 'size (W) = ({:.3f}), center (X) = ({:.3f})'.format(w, cx)
                else:
                    w = self.w * pix_sz[-1].unit
                    cx = self.cx * pix_sz[-1].unit
                    h = self.h * pix_sz[-2].unit
                    cy = self.cy * pix_sz[-2].unit
                    cmt = 'size (W, H) = ({:.3f}, {:.3f}), center (X, Y) = ({:.3f}, {:.3f})'.format(w, h, cx, cy)
            else:
                cmt = 'size (W, H) = ({:.0f}, {:.0f}), center (X, Y) = ({:.0f}, {:.0f})'.format(self.w, self.h, self.cx,
                                                                                                self.cy)
            self.update_comment(cmt)
            self.setStatusMessage('Mask ' + cmt)
            if dv.move_rect:
                QApplication.restoreOverrideCursor()
                dv.move_rect = False
        else:
            module_data = self.get_data_by_module(self.data_in, self.module)
            self.set_data_by_module(self.data_out, self.module, module_data)
            self.Outputs.data.send(self.data_out)
            # self.update_comment('no mask')
            self.setStatusMessage('No mask')
        if config_params.DEFAULT_CLOSE_WIDGETS_AFTER_APPLY:
            self.close()

    def maskScan(self, scan, mask):
        scan_mask = {}
        if self.option in [self.SUBAP_SCANS, self.SUBAP]:
            copy_items(scan, scan_mask, copydata=True)
        else:
            copy_items(scan, scan_mask)
        items = self.DATA_NAMES
        ok = True
        pix_sz = None
        if 'intensity' in scan and 'intensity' not in items:
            items += ['intensity']
        for item in items:
            if item in scan:
                if self.option in [self.SUBAP_SCANS, self.SUBAP]:
                    scan_mask[item][self.selection][~mask.astype(bool)] = np.nan
                else:
                    scan_mask[item], st, en = apply_mask(scan[item], mask)
                pix_sz = get_pix_size(scan[item], scan)
                if scan_mask[item].size == 0:
                    ok = False
                    break
        return scan_mask, pix_sz, ok
