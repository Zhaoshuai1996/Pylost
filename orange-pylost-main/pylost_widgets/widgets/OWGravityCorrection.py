import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.widget import OWWidget
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QSplitter
from PyQt5.QtWidgets import QSizePolicy as Policy
from astropy import units as u
from astropy.units import Quantity
from orangewidget.settings import Setting
from orangewidget.widget import Msg
from silx.gui.plot import Plot1D

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.ow_filereaders import AsciiReader
from pylost_widgets.util.util_functions import copy_items, integrate_slopes, differentiate_heights
from pylost_widgets.util.util_plots import OrangePlot1D
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWGravityCorrection(PylostWidgets, PylostBase):
    name = 'Gravity correction'
    description = 'Gravity correction for stitched data.'
    icon = "../icons/gravity.svg"
    priority = 23

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, default=True, auto_summary=False)
        data_gravity = Output('gravity_profile', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    density = Setting(2330.0, schema_only=True)
    young_modulus = Setting(1.3E+11, schema_only=True)
    g = Setting(9.81, schema_only=True)

    mir_length = Setting(0.0, schema_only=True)
    mir_thickness = Setting(0.0, schema_only=True)
    cyl_distance = Setting(0.0, schema_only=True)
    file_path = Setting('', schema_only=True)
    slopes_file = Setting(True, schema_only=True)
    units_file = Setting('urad', schema_only=True)

    along_y = Setting(False, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)
        self.gx = None
        self.mode = 'add'

        box = super().init_info(module=True)
        gui.button(box, self, 'Subtract gravity', callback=lambda: self.apply(mode='subtract'), autoDefault=False,
                   stretch=1, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.button(box, self, 'Add gravity (default)', callback=lambda: self.apply(mode='add'), autoDefault=False,
                   stretch=1, sizePolicy=(Policy.Fixed, Policy.Fixed))

        obox = gui.vBox(None)
        box = gui.vBox(obox, "Settings", stretch=1)
        gui.checkBox(box, self, 'along_y', 'Gravity along Y axis?')

        hbox = gui.hBox(obox)
        box = gui.vBox(hbox, "Options", stretch=1)
        gui.lineEdit(box, self, "density", "Density", labelWidth=200, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, "young_modulus", "Young modulus (elasticity)", labelWidth=200,
                     orientation=Qt.Horizontal, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, "g", "Gravity of earth", labelWidth=200, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(hbox, "Mirror options", stretch=1)
        gui.lineEdit(box, self, "mir_length", "Mirror length (mm)", labelWidth=200, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, "mir_thickness", "Mirror thickness (mm)", labelWidth=200, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, "cyl_distance", "Distance between cylinders (mm)", labelWidth=200,
                     orientation=Qt.Horizontal, sizePolicy=(Policy.Fixed, Policy.Fixed))

        fbox = gui.vBox(obox, "Load from file", stretch=1)
        box = gui.hBox(fbox)
        self.lbl_file = gui.lineEdit(box, self, 'file_path', 'Load from file: ', labelWidth=150,
                                     orientation=Qt.Horizontal)
        self.lbl_file.setEnabled(False)
        self.btnFile = gui.button(box, self, '...', callback=self.load_file, autoDefault=False, stretch=1,
                                  sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.checkBox(fbox, self, 'slopes_file', 'Check this if the file is slopes, uncheck if it is heights',
                     callback=self.check_file_data)
        gui.lineEdit(fbox, self, 'units_file', 'File data units: ', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))

        gbox = gui.vBox(None, "Gravity profile", stretch=5, sizePolicy=(Policy.MinimumExpanding, 100))
        self.plot = OrangePlot1D(parent=None)
        # self.plot.setMinimumHeight(50)
        gbox.layout().addWidget(self.plot)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(obox)
        splitter.addWidget(gbox)
        self.controlArea.layout().addWidget(splitter)

    def sizeHint(self):
        return QSize(700, 500)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True)
        if data is None:
            self.Outputs.data.send(None)
            self.Outputs.data_gravity.send(None)

    def load_data(self):
        super().load_data()
        self.apply()

    def update_comment(self, comment, prefix=''):
        # cmt = 'gravity added: ' if 'add' in self.mode else 'gravity subtracted: '
        # cmt = cmt + f'L={self.mir_length}mm, T={self.mir_thickness}mm, D={self.cyl_distance}mm'
        # cmt = cmt + f'  (Young:{self.young_modulus:.3e}, Rho={self.density})'
        super().update_comment(comment, prefix=prefix)

    def check_file_data(self):
        if self.slopes_file:
            self.units_file = 'urad'
        else:
            self.units_file = 'nm'

    def load_file(self):
        try:
            self.file_path, ext = QFileDialog.getOpenFileName(None, 'Load gravity correction')
            reader = AsciiReader(self.file_path)
            data = reader.read()
            if any(data.keys()):
                text, ok = QInputDialog.getItem(self, self.name, 'Select gravity data:', data.keys(), 0, False)
                if ok:
                    self.gx = data[text]
                    self.info.set_output_summary('Loaded file data')
                    self.plot.clear()
        except Exception as e:
            self.Error.unknown(repr(e))

    def apply(self, mode='add'):
        try:
            self.mode = mode
            self.clear_messages()
            if self.mir_length > 0 and self.mir_thickness > 0 and self.cyl_distance > 0:
                has_data = True
            elif self.file_path != '':
                has_data = True
            else:
                has_data = False
            if has_data:
                super().apply_scans()
                self.Outputs.data.send(self.data_out)
            else:
                self.info.set_output_summary('No inputs loaded')
                self.setStatusMessage('No inputs')
        except Exception as e:
            self.Error.unknown(repr(e))

    def apply_scan(self, scan, scan_name=None, comment=''):
        scan_fit = {}
        fit_comment = ''
        copy_items(scan, scan_fit)
        for i, item in enumerate(self.DATA_NAMES):
            if item in scan:
                if item == 'slopes_y':
                    continue
                Z = scan[item]
                dims = super().get_detector_dimensions(Z)
                dims_num = dims.nonzero()[0]

                axis = -2 if self.along_y and Z.ndim >= 2 else -1

                scan_fit[item] = Z
                unit = ''
                slc = [np.newaxis] * (Z.ndim)
                if isinstance(Z, MetrologyData):
                    axis_vals = Z.get_axis_val_items_detector()
                    if isinstance(axis_vals[axis], Quantity) and np.any(axis_vals[axis].value):
                        x = axis_vals[axis].to('mm').value
                        unit = 'mm'
                    else:
                        x = np.arange(Z.shape[dims_num[axis]]) * Z.pix_size_detector[axis].to('mm').value
                        unit = 'mm'
                    if len(dims_num) > (1 if axis == -2 else 0):
                        slc[dims_num[axis]] = slice(None)
                    else:
                        slc[axis] = slice(None)
                else:
                    x = np.arange(Z.shape[axis]) * self.pix_size[axis]
                    slc[axis] = slice(None)
                x = x - np.nanmean(x)
                # gx is in urad
                gx, is_slopes = self.gravity_correction(x, is_metrology_data=isinstance(Z, MetrologyData))

                if item == 'height' and is_slopes:
                    # gx in urad, x in mm
                    gx = integrate_slopes(gx, x=x, method='cumtrapz')
                    if isinstance(Z, MetrologyData):
                        gx = Quantity(gx, 'nm')
                elif item == 'slopes_x' and not is_slopes:
                    gx = differentiate_heights(gx)

                self.Outputs.data_gravity.send({item: gx})
                if Z.ndim <= 1:
                    slc = [slice(None)]
                if self.mode == 'add':
                    scan_fit[item] = scan_fit[item] + gx[slc]
                elif self.mode == 'subtract':
                    scan_fit[item] = scan_fit[item] - gx[slc]
                # fit_comment = 'gravity profile added' if self.mode == 'add' else 'gravity profile subtracted'
                fit_comment = 'gravity profile added: ' if 'add' in self.mode else 'gravity profile subtracted: '
                fit_comment = fit_comment + f'L={self.mir_length}mm, T={self.mir_thickness}mm, D={self.cyl_distance}mm'
                fit_comment = fit_comment + f' (Young:{self.young_modulus:.3e}, Rho={self.density})'
                self.plot.addCurve(x, gx.ravel(), legend='gravity')
                self.plot.setGraphXLabel('X ({})'.format(unit))
                self.plot.setGraphYLabel('Gravity {} ({})'.format(item, gx.unit if isinstance(gx, Quantity) else ''))
                # self.mainArea.show()
                self.plot.parent().show()

        return scan_fit, fit_comment

    def gravity_correction(self, x, is_metrology_data=False):
        """
        Correct gavity for a simple setup with two cylindes symmetrically supporting mirror seperated by half mirror length

        :param x: mirror x positions in mm
        :param t: thickness of mirror
        :param L: length of mirror
        :param d: seperation between cylinders, usually L/2
        :param p: for si 2340
        :param E:
        :return: gravity correction vector in urad
        """
        if self.file_path != '' and self.gx is not None:
            gx = self.gx
            if len(gx) != len(x):
                dx = np.nanmean(np.diff(x))
                xn = np.arange(len(self.gx)) * dx
                xn = xn - np.nanmean(xn) + np.min(np.abs(x))
                q = np.multiply(xn >= np.min(x), xn <= np.max(x))
                gx = self.gx[q]
            return Quantity(gx, unit=self.units_file) if is_metrology_data else gx, self.slopes_file
        else:
            p = self.density
            E = self.young_modulus
            g = self.g
            L = self.mir_length
            t = self.mir_thickness
            d = self.cyl_distance
            gx = np.full_like(x, 0.0)

            ind_i = abs(x) < (d / 2)
            ind_o = abs(x) >= (d / 2)
            S = (2 * p * g * L ** 3) * 1e-3 / (E * t ** 2)
            gx[ind_i] = S * ((x[ind_i] / L) ** 3 + (3 / 2) * (0.5 - (d / L)) * (x[ind_i] / L)) * 1e6
            gx[ind_o] = S * ((abs(x[ind_o]) / L) ** 3 - (3 / 2) * (x[ind_o] / L) ** 2 + (3 / 4) * (abs(x[ind_o]) / L) -
                    (3 / 8) * (d / L) ** 2) * np.sign(x[ind_o]) * 1e6
            return gx * u.urad if is_metrology_data else gx, True
