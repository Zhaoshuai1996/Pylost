# coding=utf-8
import numpy as np
import scipy
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QGridLayout, QSizePolicy, QSizePolicy as Policy, QTableWidgetItem
from astropy import units as u
from astropy.units import Quantity
from orangewidget.settings import Setting
from orangewidget.widget import Msg
from scipy.interpolate import griddata
from silx.gui.widgets.TableWidget import TableWidget

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import arr_to_pix, copy_items
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWInterpSubpix(PylostWidgets, PylostBase):
    name = 'Interpolate subpixels'
    description = 'Interpolate subapertures (a) to nearest pixel shift or (b) to a smaller pixel size.'
    icon = "../icons/spline.svg"
    priority = 72

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    NONE, NANS, NEW_PIXSZ, OFFSET = range(4)
    want_main_area = 0
    module = Setting('', schema_only=True)
    source = Setting(NONE, schema_only=True)
    new_pix_size_x = Setting(0.0, schema_only=True)
    new_pix_size_y = Setting(0.0, schema_only=True)
    GRIDDATA, INTERP = range(2)
    FUNC_VALS = ['griddata', 'interp2d / interp1d']
    func = Setting(GRIDDATA)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)
        gui.button(box, self, 'Interpolate', callback=self.apply, autoDefault=False, stretch=1,
                   sizePolicy=Policy(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(self.controlArea, "Interpolate options", stretch=9)
        cb = gui.comboBox(box, self, 'func', label='Scipy algorithm used for interpolation :', box=False,
                          orientation=Qt.Horizontal, items=self.FUNC_VALS,
                          sizePolicy=QSizePolicy(Policy.Fixed, Policy.Fixed))
        layout = QGridLayout()
        gui.widgetBox(box, margin=0, orientation=layout, addSpace=True)
        rbox = gui.radioButtons(None, self, "source", box=True, addSpace=True, addToLayout=False)

        rb_button = gui.appendRadioButton(rbox, "No interpolation", addToLayout=False)
        layout.addWidget(rb_button, 1, 0, Qt.AlignVCenter)
        rb_button = gui.appendRadioButton(rbox, "Interpolate 'nan' and 'inf' pixels", addToLayout=False)
        layout.addWidget(rb_button, 2, 0, Qt.AlignVCenter)

        rb_button = gui.appendRadioButton(rbox, "Interpolate to new pixel size:", addToLayout=False)
        layout.addWidget(rb_button, 3, 0, Qt.AlignVCenter)
        box = gui.hBox(None)
        gui.lineEdit(box, self, 'new_pix_size_x', 'x (mm)', labelWidth=35, orientation=Qt.Horizontal,
                     callback=self.change_psz)
        layout.addWidget(box, 3, 1, Qt.AlignVCenter)
        box = gui.hBox(None)
        gui.lineEdit(box, self, 'new_pix_size_y', 'y (mm)', labelWidth=35, orientation=Qt.Horizontal,
                     callback=self.change_psz)
        layout.addWidget(box, 3, 2, Qt.AlignVCenter)

        rb_button = gui.appendRadioButton(rbox,
                                          'Interpolation of subpixel offsets (shown below) from non-integer-pixel translations of motors',
                                          addToLayout=False)
        layout.addWidget(rb_button, 4, 0, 1, 3)
        self.tblOffsets = TableWidget(None)
        layout.addWidget(self.tblOffsets, 5, 0, 1, 3)

    def sizeHint(self):
        return QSize(500, 500)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True)
        if data is None:
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        super().load_data()
        # self.apply()

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Interpolation of data')

    def change_psz(self):
        if (self.new_pix_size_x > 0 or self.new_pix_size_y > 0):
            self.source = self.NEW_PIXSZ

    def apply(self):
        try:
            super().apply_scans()
            self.Outputs.data.send(self.data_out)
        except Exception as e:
            self.Outputs.data.send(None)
            self.Error.unknown(repr(e))

    def apply_scan(self, scan, scan_name=None, comment=''):
        scan_fit = {}
        copy_items(scan, scan_fit)
        for i, item in enumerate(self.DATA_NAMES):
            if item in scan:
                Z = scan[item]
                dims = self.get_detector_dimensions(Z)
                axes = dims.nonzero()[0][::-1]

                scan_fit[item] = Z
                if self.source == self.OFFSET:
                    scan_fit, comment = self.interp_offsets(item, scan, scan_fit, Z, dims)
                elif self.source == self.NEW_PIXSZ:
                    if (self.new_pix_size_x > 0 or self.new_pix_size_y > 0):
                        scan_fit, comment = self.interp_psz(item, scan, scan_fit, Z, dims)
                    else:
                        self.Error.unknown('Please enter new pixel size x or y')
                elif self.source == self.NANS:
                    self.new_pix_size_x = self.new_pix_size_y = 0
                    scan_fit, comment = self.interp_psz(item, scan, scan_fit, Z, dims)
                    comment = 'Interpolated nan and inf values'

        return scan_fit, comment

    def interp_psz(self, item, scan, scan_fit, Z, dims):
        pix_sz = scan['pix_size'] if 'pix_size' in scan else [1] * np.sum(dims)
        if isinstance(Z, MetrologyData):
            pix_sz = Z.pix_size_detector
        pix_sz_new = pix_sz
        pix_sz_values = [x.to('mm').value if isinstance(x, Quantity) else x for x in pix_sz]

        shp = tuple(np.asarray(Z.shape)[dims])
        xm = shp[-1] * pix_sz_values[-1]
        xnew = x = np.arange(shp[-1]) * pix_sz_values[-1]
        if self.new_pix_size_x > 0:
            xnew = np.arange(0, xm, self.new_pix_size_x)
            pix_sz_new[-1] = self.new_pix_size_x * u.mm if isinstance(Z, MetrologyData) else self.new_pix_size_x
        is2d = False
        if np.any(dims) and False in dims:
            if not np.sum(dims) in [1, 2]:  # only 1D or 2D
                raise Exception('Interpolation is implemented for 1D and 2D only')
            if np.sum(dims) == 2:
                is2d = True
        elif Z.ndim >= 2:
            is2d = True
        if is2d:
            ym = shp[-2] * pix_sz_values[-2]
            ynew = y = np.arange(shp[-2]) * pix_sz_values[-2]
            if self.new_pix_size_y > 0:
                ynew = np.arange(0, ym, self.new_pix_size_y)
                pix_sz_new[-2] = self.new_pix_size_y * u.mm if isinstance(Z, MetrologyData) else self.new_pix_size_y

        shp_zf = np.array(Z.shape)
        if np.any(dims) and False in dims:  # Apply interpolate for detector dimensions.
            shp_zf[dims] = [len(ynew), len(xnew)] if np.sum(dims) == 2 else len(xnew)
            Zf = np.full(shp_zf, np.nan, dtype=float)
            shp_nd = tuple(np.asarray(Z.shape)[np.invert(dims)])
            idx_full = np.asarray([slice(None)] * Z.ndim)
            for idx in np.ndindex(shp_nd):
                idx_full[np.invert(dims)] = np.asarray(idx)
                zi = Z[tuple(idx_full)]
                if self.new_pix_size_x == 0 and self.new_pix_size_y == 0 and np.all(
                        np.isfinite(zi)):  # Interpolate only if nans or infs exist
                    Zf[tuple(idx_full)] = zi
                    continue
                if np.sum(dims) == 2:  # 2D surfaces
                    Zf[tuple(idx_full)] = self.interp2D_psz(zi, x, y, xnew, ynew)
                else:  # 1D lines
                    Zf[tuple(idx_full)] = self.interp1D_psz(zi, x, xnew)
        else:
            if Z.ndim == 1:
                if self.new_pix_size_x == 0 and self.new_pix_size_y == 0 and np.all(
                        np.isfinite(Z)):  # Interpolate only if nans or infs exist
                    Zf = Z
                else:
                    Zf = self.interp1D_psz(Z, x, xnew)
            elif Z.ndim == 2:
                if self.new_pix_size_x == 0 and self.new_pix_size_y == 0 and np.all(
                        np.isfinite(Z)):  # Interpolate only if nans or infs exist
                    Zf = Z
                else:
                    Zf = self.interp2D_psz(Z, x, y, xnew, ynew)
            else:
                shp_zf = Z.shape[:-2] + (len(ynew), len(xnew))
                Zf = np.full(shp_zf, np.nan, dtype=float)
                for idx in np.ndindex(Z.shape[:-2]):
                    if self.new_pix_size_x == 0 and self.new_pix_size_y == 0 and np.all(
                            np.isfinite(Z[idx])):  # Interpolate only if nans or infs exist
                        Zf[idx] = Z[idx]
                        continue
                    Zf[idx] = self.interp2D_psz(Z[idx], x, y, xnew, ynew)

        if isinstance(Z, MetrologyData):
            Zf = Z.copy_to(Zf)
            Zf._copy_items()
            Zf._set_pix_size(pix_sz_new)

        comment = 'Interpolated to new pixel size: {}'.format(pix_sz_new)
        scan_fit[item] = Zf
        return scan_fit, comment

    def interp_offsets(self, item, scan, scan_fit, Z, dims):
        comment = ''
        mx = np.array(scan['motor_X']) if 'motor_X' in scan else []
        my = np.array(scan['motor_Y']) if 'motor_Y' in scan else []
        pix_sz = scan['pix_size'] if 'pix_size' in scan else []
        if isinstance(Z, MetrologyData) and np.any(Z.motors):
            pix_sz = Z.pix_size_detector
            for m in Z.motors:
                mx = Quantity(np.array(m['values']), unit=m['unit']) if m['name'] == 'motor_X' else mx
                my = Quantity(np.array(m['values']), unit=m['unit']) if m['name'] == 'motor_Y' else my

        if len(mx) > 0 and len(pix_sz) > 0:
            Zf = np.full_like(Z.value if isinstance(Z, MetrologyData) else Z, np.nan)
            if len(my) == 0:
                my = np.zeros_like(mx)

            ox, xoff = arr_to_pix(mx, pix_sz[-1])
            oy, yoff = arr_to_pix(my, pix_sz[-2 if len(pix_sz) > 1 else -1])
            if np.any(dims) and False in dims:  # Apply interpolate for detector dimensions.
                if not np.sum(dims) in [1, 2]:  # only 1D or 2D
                    raise Exception('Interpolation is implemented for 1D and 2D only')
                shp_nd = tuple(np.asarray(Z.shape)[np.invert(dims)])
                idx_full = np.asarray([slice(None)] * Z.ndim)
                for idx in np.ndindex(shp_nd):
                    idx_full[np.invert(dims)] = np.asarray(idx)
                    zi = Z[tuple(idx_full)]
                    if np.sum(dims) == 2:  # 2D surfaces
                        Zf[tuple(idx_full)] = self.interp2D_offset(zi, xoff[idx], yoff[idx])
                    else:  # 1D lines
                        Zf[tuple(idx_full)] = self.interp1D_offset(zi, xoff[idx])
            else:
                if Z.ndim == 2:
                    for i in np.arange(Z.shape[0]):
                        Zf[i] = self.interp1D_offset(Z[i], xoff[i])
                elif Z.ndim == 3:
                    for i in np.arange(Z.shape[0]):
                        Zf[i] = self.interp2D_offset(Z[i], xoff[i], yoff[i])
                else:
                    for idx in np.ndindex(Z.shape[:-2]):
                        Zf[idx] = self.interp2D_offset(Z[idx], xoff[idx], yoff[idx])
            omx = np.min(mx) + ox * pix_sz[-1]
            omy = np.min(my) + oy * pix_sz[-2 if len(pix_sz) > 1 else -1]
            if isinstance(Z, MetrologyData):
                Zf = Z.copy_to(Zf)
                Zf._copy_items()
                Zf.update_motor('motor_X', omx.value, omx.unit)
                Zf.update_motor('motor_Y', omy.value, omy.unit)
            else:
                scan_fit['motor_X'] = omx
                scan_fit['motor_Y'] = omy
            scan_fit[item] = Zf
            self.fillTable(xoff.ravel(), yoff.ravel())

            comment = 'Interpolated motor offsets(pix) \nX: max={:.3f}, min={:.3f} ' \
                      '\nY: max={:.3f}, min={:.3f}'.format(np.nanmax(xoff), np.nanmin(xoff),
                                                           np.nanmax(yoff), np.nanmin(yoff))
        return scan_fit, comment

    @staticmethod
    def interp1D_psz(zi, x, xn):
        zn = np.nan
        if np.any(xn) and np.any(np.isfinite(zi)):
            q = np.isfinite(zi)
            f = scipy.interpolate.interp1d(x[q], zi[q])
            zn = f(xn)
        return zn

    @staticmethod
    def interp1D_offset(zi, xi_off):
        zn = zi
        if xi_off and np.any(np.isfinite(zi)):
            x = np.arange(zi.shape[0])
            q = np.isfinite(zi)
            f = scipy.interpolate.interp1d(x[q], zi[q])
            xn = x - xi_off
            zn = f(xn)
        return zn

    def interp2D_psz(self, zi, x, y, xn, yn):
        zn = np.nan
        if np.any(xn) and np.any(yn) and np.any(np.isfinite(zi)):
            xx, yy = np.meshgrid(x, y)
            q = np.isfinite(zi)
            if self.func == self.INTERP:
                f = scipy.interpolate.interp2d(xx[q], yy[q], zi[q], kind='cubic')
                zn = f(xn, yn)
            elif self.func == self.GRIDDATA:
                xxn, yyn = np.meshgrid(xn, yn)
                zn = griddata((xx[q], yy[q]), zi[q], (xxn, yyn), method='cubic')
        return zn

    def interp2D_offset(self, zi, xi_off, yi_off):
        zn = zi
        if (xi_off or yi_off) and np.any(np.isfinite(zi)):
            x = np.arange(zi.shape[1])
            y = np.arange(zi.shape[0])
            xx, yy = np.meshgrid(x, y)
            xn = x - xi_off
            yn = y - yi_off
            q = np.isfinite(zi)
            if self.func == self.INTERP:
                f = scipy.interpolate.interp2d(xx[q], yy[q], zi[q])
                zn = f(xn, yn)
            elif self.func == self.GRIDDATA:
                xxn, yyn = np.meshgrid(xn, yn)
                zn = griddata((xx[q], yy[q]), zi[q], (xxn, yyn), method='cubic')
        return zn

    def fillTable(self, x, y):
        arr = [x, y]
        cols = 2
        rows = len(x)
        self.tblOffsets.setRowCount(rows)
        self.tblOffsets.setColumnCount(cols)
        self.tblOffsets.setHorizontalHeaderLabels("X ;Y ".split(";"))

        for i in np.arange(rows):
            for j in np.arange(cols):
                val = QTableWidgetItem('{:.3f}'.format(arr[j][i]))
                val.setToolTip('{}'.format(arr[j][i]))
                val.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.tblOffsets.setItem(i, j, val)
