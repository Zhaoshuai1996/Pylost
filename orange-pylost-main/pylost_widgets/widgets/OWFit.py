# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QGridLayout, QScrollArea, QSizePolicy as Policy, QWidget
from astropy import units as u
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from PyLOSt.algorithms.util.util_math import nbTerms
from pylost_widgets.config import config_params
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.base.EllipseBase import EllipseBase
from pylost_widgets.util.math import rmse
from pylost_widgets.util.util_functions import MODULE_MULTI, MODULE_SINGLE, copy_items, fit_nD_metrology, get_dict_item
from pylost_widgets.util.util_scripts import import_paths
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWFit(PylostWidgets, PylostBase):
    """Widget to fit datasets a polynomial or ellise and subtract the fit"""
    name = 'Fit'
    description = 'Fit raw or stitched data.'
    icon = "../icons/fit.svg"
    priority = 22

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, default=True, auto_summary=False)
        fit = Output('fit', dict, auto_summary=False)
        coef = Output('coefficients', dict, auto_summary=False)

    want_main_area = 0

    DEFAULT, TERMS, POLY, ELLIPSE = range(4)
    poly_items = {'POLY': 'Polynomial', 'LEGENDRE': 'Legendre polynomial', 'ZERNIKE': 'Zernike polynomial'}

    module = Setting('', schema_only=True)
    source = Setting(DEFAULT, schema_only=True)

    ellipse_scripts = {'EllipsePylost': ''}
    sel_ellipse_script = Setting(list(ellipse_scripts.keys())[0], schema_only=True)

    piston = Setting(True, schema_only=True)
    tilt = Setting(True, schema_only=True)
    tiltX = Setting(False, schema_only=True)
    tiltY = Setting(False, schema_only=True)
    sphere = Setting(False, schema_only=True)
    cylX = Setting(False, schema_only=True)
    cylY = Setting(False, schema_only=True)
    ast = Setting(False, schema_only=True)
    sel_poly = Setting(poly_items['POLY'], schema_only=True)
    poly_deg = Setting(0, schema_only=True)
    poly_deg_start = Setting(0, schema_only=True)
    ellipse_p = Setting(0.0, schema_only=True)
    ellipse_q = Setting(0.0, schema_only=True)
    ellipse_theta = Setting(0.0, schema_only=True)
    ellipse_center_offset = Setting(0.0, schema_only=True)
    ellipse_rotate = Setting(0.0, schema_only=True)
    ellipse_piston = Setting(0.0, schema_only=True)
    ellipse_p_check = Setting(False, schema_only=True)
    ellipse_q_check = Setting(True, schema_only=True)
    ellipse_theta_check = Setting(True, schema_only=True)
    ellipse_center_offset_check = Setting(False, schema_only=True)
    ellipse_rotate_check = Setting(True, schema_only=True)
    ellipse_piston_check = Setting(True, schema_only=True)
    fit_all = Setting(True, schema_only=True)
    fit_height = Setting(False, schema_only=True)
    fit_slopes = Setting(False, schema_only=True)
    fit_rotation = Setting(False, schema_only=True)
    center = Setting(False, schema_only=True)
    rescale = Setting(False, schema_only=True)
    AUTO, MANUAL = range(2)
    rescale_type = Setting(AUTO, schema_only=True)
    bin_fit = Setting(False, schema_only=True)
    bin_y = Setting(1, schema_only=True)
    bin_x = Setting(1, schema_only=True)

    interpolate = Setting(True, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    # node = None
    # scheme = None

    # def __new__(cls, *args, captionTitle=None, **kwargs):
    #     klass = super().__new__(cls, *args, captionTitle=cls.name, **kwargs)
    #     manager = kwargs.get('signal_manager', None)
    #     if manager is not None:
    #         cls.node = list(manager._SignalManager__node_outputs.keys())[0]
    #         node = cls.node
    #         cls.scheme = manager._SignalManager__workflow
    #         scheme = cls.scheme
    #         nodes = scheme._Scheme__nodes
    #         for node in nodes:
    #             if node.title.startswith(cls.name):
    #                 node.title = cls.name
    #     return klass
    #

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)
        self.fits = {}
        self.filter_terms = [0, 0, 0, 0, 0, 0, 0]

        box = super().init_info(module=True)
        self.btnApply = gui.button(box, self, 'Subtract fit', callback=self.applyFit, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, "Fit shape", margin=10, orientation=layout, addSpace=True, stretch=8)
        vbox = gui.radioButtons(None, self, "source", callback=self.change_source, box=True, addSpace=True,
                                addToLayout=False)

        rb1 = gui.appendRadioButton(vbox, "None", addToLayout=False)
        layout.addWidget(rb1, 0, 0, Qt.AlignVCenter)

        rb2 = gui.appendRadioButton(vbox, "Terms", addToLayout=False)
        layout.addWidget(rb2, 1, 0, Qt.AlignVCenter)

        gridTerms = QGridLayout()
        ch0 = gui.checkBox(None, self, "piston", "Piston", callback=self.filter)
        ch1 = gui.checkBox(None, self, "tilt", "Tilt (XY)", callback=[self.sel_tilt, self.filter])
        ch11 = gui.checkBox(None, self, "tiltX", "Tilt X", callback=[self.sel_tilt_x, self.filter])
        ch12 = gui.checkBox(None, self, "tiltY", "Tilt Y", callback=[self.sel_tilt_y, self.filter])
        ch2 = gui.checkBox(None, self, "sphere", "Sphere", callback=[self.sel_sph, self.filter])
        ch21 = gui.checkBox(None, self, "cylX", "Cylinder X", callback=[self.sel_cyl_x, self.filter])
        ch22 = gui.checkBox(None, self, "cylY", "Cylinder Y", callback=[self.sel_cyl_y, self.filter])
        ch23 = gui.checkBox(None, self, "ast", "Astigmatism", callback=self.filter)
        gridTerms.addWidget(ch0, 0, 0, Qt.AlignVCenter)
        gridTerms.addWidget(ch1, 1, 0, Qt.AlignVCenter)
        gridTerms.addWidget(ch11, 1, 1, Qt.AlignVCenter)
        gridTerms.addWidget(ch12, 1, 2, Qt.AlignVCenter)
        gridTerms.addWidget(ch2, 2, 0, Qt.AlignVCenter)
        gridTerms.addWidget(ch21, 2, 1, Qt.AlignVCenter)
        gridTerms.addWidget(ch22, 2, 2, Qt.AlignVCenter)
        gridTerms.addWidget(ch23, 3, 0, Qt.AlignVCenter)
        grid = QWidget(None)
        grid.setLayout(gridTerms)
        layout.addWidget(grid, 1, 1, 1, 3)

        rb3 = gui.appendRadioButton(vbox, "Polynomial", addToLayout=False)
        layout.addWidget(rb3, 2, 0, Qt.AlignVCenter)

        box = gui.hBox(None)
        self.poly_combo = gui.comboBox(box, self, "sel_poly", sendSelectedValue=True,
                                       items=list(self.poly_items.values()), sizePolicy=(Policy.Fixed, Policy.Fixed),
                                       callback=lambda: setattr(self, 'source', self.POLY))
        self.polyDegree = gui.lineEdit(box, self, 'poly_deg', "Degree:", labelWidth=50, orientation=Qt.Horizontal,
                                       sizePolicy=(Policy.Fixed, Policy.Fixed),
                                       callback=lambda: setattr(self, 'source', self.POLY))
        self.polyDegStart = gui.lineEdit(box, self, 'poly_deg_start', "Start degree:", labelWidth=75,
                                         orientation=Qt.Horizontal, sizePolicy=(Policy.Fixed, Policy.Fixed),
                                         callback=lambda: setattr(self, 'source', self.POLY))

        layout.addWidget(box, 2, 1, 1, 2)

        hbox = gui.hBox(None)
        rb4 = gui.appendRadioButton(vbox, "Ellipse", addToLayout=False)
        hbox.layout().addWidget(rb4)
        self.import_ellipse_scripts()
        gui.comboBox(hbox, self, 'sel_ellipse_script', sendSelectedValue=True, items=list(self.ellipse_scripts.keys()),
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=lambda: setattr(self, 'source', self.ELLIPSE))
        layout.addWidget(hbox, 3, 0, Qt.AlignVCenter)

        gridTerms = QGridLayout()
        box = gui.hBox(None)
        gui.checkBox(box, self, "ellipse_p_check", "P (m): ")
        self.elP = gui.lineEdit(box, self, 'ellipse_p', "", orientation=Qt.Horizontal,
                                sizePolicy=Policy(Policy.Fixed, Policy.Fixed),
                                callback=lambda: setattr(self, 'source', self.ELLIPSE))
        gridTerms.addWidget(box, 0, 0, Qt.AlignVCenter)
        box = gui.hBox(None)
        gui.checkBox(box, self, "ellipse_q_check", "Q (m): ")
        self.elQ = gui.lineEdit(box, self, 'ellipse_q', "", orientation=Qt.Horizontal,
                                sizePolicy=Policy(Policy.Fixed, Policy.Fixed),
                                callback=lambda: setattr(self, 'source', self.ELLIPSE))
        gridTerms.addWidget(box, 0, 1, Qt.AlignVCenter)
        box = gui.hBox(None)
        gui.checkBox(box, self, "ellipse_theta_check", "Theta (mrad): ")
        self.elTh = gui.lineEdit(box, self, 'ellipse_theta', "", orientation=Qt.Horizontal,
                                 sizePolicy=(Policy.Fixed, Policy.Fixed),
                                 callback=lambda: setattr(self, 'source', self.ELLIPSE))
        # layout.addWidget(box, 3, 1, 1, 4)
        gridTerms.addWidget(box, 0, 2, Qt.AlignVCenter)

        box = gui.hBox(None)
        gui.checkBox(box, self, "ellipse_center_offset_check", "Center offset (mm): ")
        self.elCO = gui.lineEdit(box, self, 'ellipse_center_offset', "", orientation=Qt.Horizontal,
                                 sizePolicy=(Policy.Fixed, Policy.Fixed),
                                 callback=lambda: setattr(self, 'source', self.ELLIPSE))
        gridTerms.addWidget(box, 1, 0, Qt.AlignVCenter)
        box = gui.hBox(None)
        gui.checkBox(box, self, "ellipse_rotate_check", "Rotate (mrad): ")
        self.elR = gui.lineEdit(box, self, 'ellipse_rotate', "", orientation=Qt.Horizontal,
                                sizePolicy=(Policy.Fixed, Policy.Fixed),
                                callback=lambda: setattr(self, 'source', self.ELLIPSE))
        gridTerms.addWidget(box, 1, 1, Qt.AlignVCenter)
        box = gui.hBox(None)
        gui.checkBox(box, self, "ellipse_piston_check", "Piston (nm): ")
        self.elPt = gui.lineEdit(box, self, 'ellipse_piston', "", orientation=Qt.Horizontal,
                                 sizePolicy=(Policy.Fixed, Policy.Fixed),
                                 callback=lambda: setattr(self, 'source', self.ELLIPSE))
        # layout.addWidget(box, 4, 1, 1, 4)
        gridTerms.addWidget(box, 1, 2, Qt.AlignVCenter)
        grid = QWidget(None)
        grid.setLayout(gridTerms)
        layout.addWidget(grid, 3, 1, 2, 4)
        layout.setVerticalSpacing(15)

        box = gui.hBox(None)
        layout.addWidget(box, 5, 1, 1, 4)
        layout.setVerticalSpacing(15)

        layout = QGridLayout()
        row = 0
        gui.widgetBox(self.controlArea, "Additional Options", margin=10, orientation=layout, addSpace=True, stretch=1)
        cb00 = gui.checkBox(None, self, "fit_all",
                            "Fit both heights and slopes independently (integrate/diferentiate if not existing)",
                            callback=self.change_fit_all)
        layout.addWidget(cb00, row, 0, 1, 4)
        row += 1
        cb0 = gui.checkBox(None, self, "fit_height", "Fit only heights", callback=self.change_fit_height)
        cb1 = gui.checkBox(None, self, "fit_slopes", "Fit only slopes", callback=self.change_fit_slopes)
        layout.addWidget(cb0, row, 0, Qt.AlignVCenter)
        layout.addWidget(cb1, row, 1, Qt.AlignVCenter)
        row += 1
        cb2 = gui.checkBox(None, self, "center", "Center data (only for fitting)")
        layout.addWidget(cb2, row, 0)
        cb22 = gui.checkBox(None, self, "fit_rotation", "Auto rotate")
        layout.addWidget(cb22, row, 1, Qt.AlignVCenter)
        row += 1
        cb2 = gui.checkBox(None, self, "rescale", "Rescale units (only for fitting)", callback=self.change_rescale)
        layout.addWidget(cb2, row, 0, 1, 4)
        row += 1
        rbox = gui.radioButtons(None, self, "rescale_type", box=True, addSpace=True, addToLayout=False)
        self.rb_auto = gui.appendRadioButton(rbox, "Auto scale", addToLayout=False)
        layout.addWidget(self.rb_auto, row, 0, Qt.AlignVCenter)
        self.rb_manual = gui.appendRadioButton(rbox, "Manual", addToLayout=False)
        layout.addWidget(self.rb_manual, row, 1, Qt.AlignVCenter)
        self.change_rescale()
        row += 1
        box = gui.hBox(None, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.checkBox(box, self, "bin_fit", "Bin data (only for fitting)")
        gui.lineEdit(box, self, 'bin_x', "X: ", orientation=Qt.Horizontal)
        gui.lineEdit(box, self, 'bin_y', "Y: ", orientation=Qt.Horizontal)
        layout.addWidget(box, row, 0, 1, 4)

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, "Fit results", margin=10, orientation=layout, addSpace=True, stretch=1)
        lbl1 = gui.widgetLabel(None, "Fit parameters ::  ")
        layout.addWidget(lbl1, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.infoFit = gui.widgetLabel(None, "")
        scroll.setWidget(self.infoFit)
        layout.addWidget(scroll, 0, 1)
        lbl1 = gui.widgetLabel(None, "Shape parameters ::  ")
        layout.addWidget(lbl1, 1, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.infoParms = gui.widgetLabel(None, "")
        scroll.setWidget(self.infoParms)
        layout.addWidget(scroll, 1, 1)

    def sizeHint(self):
        return QSize(500, 50)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True, show_all_default_names=True)
        if data is not None:
            self.enable_options()
        else:
            self.disable_options()
            self.Outputs.data.send(None)
            self.Outputs.fit.send(None)
            self.Outputs.coef.send(None)

    def load_data(self, multi=False):
        super().load_data()
        self.update_bin_xy()
        self.update_filter()
        self.applyFit()

    def import_ellipse_scripts(self):
        """Import ellipse scripts from external folder"""
        import_paths(param_name='GENERAL_SCRIPT_PATH')
        all_classes = EllipseBase.__subclasses__()
        for cls in all_classes:
            self.ellipse_scripts[cls.__name__] = cls

    def update_bin_xy(self):
        """Update binning if entered"""
        try:
            for item in self.DEFAULT_DATA_NAMES:
                data = get_dict_item(self.data_in, item)
                if data is not None:
                    if data.ndim > 0 and self.bin_x == 1:
                        self.bin_x = 1 + (data.shape[-1] // 500)
                    if data.ndim > 1 and self.bin_y == 1:
                        self.bin_y = 1 + (data.shape[-2] // 500)
                    break
        except Exception:
            pass

    def change_source(self):
        """Callback after chanign source e.g. polynomial or ellipse or filter terms"""
        if self.source == self.DEFAULT:
            copy_items(self.data_in, self.data_out, deepcopy=True)
            self.Outputs.data.send(self.data_out)

    def change_rescale(self):
        """REscale is selected/unselected"""
        if self.rescale:
            self.rb_auto.setEnabled(True)
            self.rb_manual.setEnabled(True)
        else:
            self.rb_auto.setEnabled(False)
            self.rb_manual.setEnabled(False)

    def change_fit_all(self):
        """Fit both slopes and heights, checkbox"""
        self.data_out = {}
        copy_items(self.data_in, self.data_out, deepcopy=True)
        self.Outputs.data.send(None)

    def change_fit_slopes(self):
        """Fit only slopes"""
        if self.fit_slopes:
            self.fit_height = False

    def change_fit_height(self):
        """Fit only heights"""
        if self.fit_height:
            self.fit_slopes = False

    def update_fit_info(self, txt):
        """Update fit information in the UI"""
        tmp = self.infoFit.text() + '\n' if self.infoFit.text() != '' else ''
        self.infoFit.setText(tmp + txt)

    def update_param_info(self, txt):
        """Update parameters information in the UI"""
        tmp = self.infoParms.text() + '\n' if self.infoParms.text() != '' else ''
        self.infoParms.setText(tmp + txt)

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Applied fit')

    def applyFit(self):
        """Apply fit"""
        try:
            self.setStatusMessage('')
            self.clear_messages()
            self.infoFit.setText('')
            self.infoParms.setText('')
            self.info.set_output_summary('Fitting...')
            QtWidgets.qApp.processEvents()
            if self.source == self.DEFAULT:
                copy_items(self.data_in, self.data_out, deepcopy=True)
                self.Outputs.data.send(self.data_out)
                self.info.set_output_summary('')
            else:
                # profiler = start_profiler()
                fit_comment = ''
                self.more_opts = {'center_data': self.center, 'rescale': self.rescale,
                                  'rescale_type': self.rescale_type,
                                  'bin_fit': self.bin_fit, 'bin_x': self.bin_x, 'bin_y': self.bin_y,
                                  'ellipse_rotation': self.fit_rotation}
                self.coef_scans = {}
                fits = {}
                self.fits = {}
                module_data = self.get_data_by_module(self.data_in, self.module)
                scan_input = {}
                if self.module in MODULE_MULTI:
                    scans_fit = {}
                    for it in module_data:
                        scan = module_data[it]
                        scan_input = self.full_dataset(scan) if self.fit_all else scan
                        scan_fit, coef_scan, fit, fit_comment = self.applyFit_scan(scan_input, scan_name=it)
                        self.coef_scans[it] = coef_scan
                        scans_fit[it] = scan_fit
                        fits[it] = fit
                    self.set_data_by_module(self.data_out, self.module, scans_fit)
                elif self.module in MODULE_SINGLE:
                    scan_input = self.full_dataset(module_data) if self.fit_all else module_data
                    scan_fit, coef_scan, fit, fit_comment = self.applyFit_scan(scan_input)
                    self.coef_scans = coef_scan
                    fits = fit
                    self.set_data_by_module(self.data_out, self.module, scan_fit)
                self.update_comment(fit_comment)
                self.setStatusMessage('{}'.format(fit_comment))
                self.data_out['coef_scans'] = self.coef_scans
                self.Outputs.data.send(self.data_out)

                self.set_data_by_module(self.fits, self.module, fits)
                self.Outputs.fit.send(self.fits)
                self.Outputs.coef.send(self.coef_scans)
                self.info.set_output_summary(fit_comment)
                # print_profiler(profiler)
            if config_params.DEFAULT_CLOSE_WIDGETS_AFTER_APPLY:
                self.close()
        except Exception as e:
            print(e)
            self.Error.unknown(repr(e))

    def applyFit_scan(self, scan, scan_name=''):
        """Apply fit for reach scan"""
        coef_scan = {}
        scan_fit = {}
        fit = {}
        fit_comment = ''
        copy_items(scan, scan_fit, deepcopy=True)
        for i, item in enumerate(self.DATA_NAMES):
            if item in scan:
                if self.fit_height and item != 'height':
                    continue
                if self.fit_slopes and item not in ['slopes_x', 'slopes_y']:
                    continue
                if self.source == self.POLY:
                    enDeg = self.poly_deg
                    stDeg = self.poly_deg_start
                    if self.fit_all:
                        if item in ['slopes_x', 'slopes_y']:
                            enDeg = enDeg - 1 if enDeg > 0 else enDeg
                            stDeg = stDeg - 1 if stDeg > 0 else stDeg
                    typ = 'poly'
                    fit_comment += '{}: '.format(item)
                    if self.sel_poly == self.poly_items['ZERNIKE']:
                        typ = 'zernike'
                        fit_comment = 'zernike '
                    if self.sel_poly == self.poly_items['LEGENDRE']:
                        typ = 'legendre'
                        fit_comment = 'legendre '
                    coef_scan[item], scan_fit[item], fit[item] = fit_nD_metrology(scan[item], pix_size=self.pix_size,
                                                                                  degree=enDeg, start_degree=stDeg,
                                                                                  dtyp=item, more_opts=self.more_opts,
                                                                                  typ=typ)
                    fit_comment += 'polynomial with degree {} {}\n'.format(enDeg, ', and starting degree {}'.format(
                        stDeg) if stDeg > 0 else '')
                    self.update_info_poly(item, scan_fit, coef_scan, scan, enDeg, stDeg)
                elif self.source == self.TERMS:
                    coef_scan[item], scan_fit[item], fit[item] = fit_nD_metrology(scan[item], pix_size=self.pix_size,
                                                                                  filter_terms_poly=self.filter_terms,
                                                                                  dtyp=item, more_opts=self.more_opts)
                    fit_comment = 'terms {} removed'.format(self.get_filter_terms())
                    self.update_info_terms(item, scan_fit, coef_scan, scan)
                elif self.source == self.ELLIPSE:
                    if item not in ['slopes_x', 'height']:
                        continue
                    ep = [self.ellipse_p, self.ellipse_q, self.ellipse_theta, self.ellipse_center_offset,
                          self.ellipse_rotate, self.ellipse_piston]
                    ep_check = [self.ellipse_p_check, self.ellipse_q_check, self.ellipse_theta_check,
                                self.ellipse_center_offset_check, self.ellipse_rotate_check, self.ellipse_piston_check]
                    if item == 'slopes_x':
                        ep_check[-1] = False  # Always disable piston for slopes fitting

                    # Remove a polynomial terms piston, tilts before and after ellipse fit
                    deg = 1 if item in ['height'] else 0
                    _, scan_fit[item], _ = fit_nD_metrology(scan[item], pix_size=self.pix_size, degree=deg, dtyp=item,
                                                            more_opts=self.more_opts)
                    coef_scan[item], scan_fit[item], fit[item] = fit_nD_metrology(scan_fit[item],
                                                                                  pix_size=self.pix_size, elp_params=ep,
                                                                                  useMaskForXY=False,
                                                                                  elp_params_check=ep_check,
                                                                                  typ='ellipse', dtyp=item,
                                                                                  more_opts=self.more_opts,
                                                                                  sel_script=self.sel_ellipse_script,
                                                                                  sel_script_class=self.ellipse_scripts[
                                                                                      self.sel_ellipse_script])
                    _, scan_fit[item], _ = fit_nD_metrology(scan_fit[item], pix_size=self.pix_size, degree=deg,
                                                            dtyp=item, more_opts=self.more_opts)

                    if isinstance(coef_scan[item], np.ndarray) and coef_scan[item].ndim > 1:
                        slc = tuple([slice(0)] * (coef_scan[item].ndim - 1) + [slice(None)])
                        cd = coef_scan[item][slc]
                    else:
                        cd = coef_scan[item]

                    rc_tan = (2 / np.sin(cd[2] * 1e-3)) * (cd[0] * cd[1]) / (cd[0] + cd[1])
                    scan_fit['{}_Rc_tan'.format(item)] = rc_tan * u.m
                    if self.ellipse_p_check:
                        scan_fit['{}_p'.format(item)] = cd[0] * u.m
                    if self.ellipse_q_check:
                        scan_fit['{}_q'.format(item)] = cd[1] * u.m
                    if self.ellipse_theta_check:
                        scan_fit['{}_theta'.format(item)] = cd[2] * u.mrad
                    if ('height' not in self.DATA_NAMES) or ('height' in self.DATA_NAMES and item == 'height'):
                        fit_comment = 'ellipse (script:{}) with best fit (given) params for {} : p = {:.7f} ({:.3f}) m, q = {:.7f} ({:.3f}) m, theta = {:.7f} ({:.3f}) mrad \n' \
                            .format(self.sel_ellipse_script, item, cd[0], ep[0], cd[1], ep[1], cd[2], ep[2])
                    self.update_fit_info(
                        'best fit ellipse for {} : Rmse residuals {:.3f}\n [p (m), q (m), theta (mrad), center offset (mm), tilt (mrad), piston(nm), rotation (deg)] = {} \n'.format(item, rmse(scan_fit[item]), self._format_coef(coef_scan[item])))

        return scan_fit, coef_scan, fit, fit_comment

    def update_info_poly(self, item, scan_fit, coef_scan, scan, enDeg, stDeg):
        """Update info box after polynomial fit, e.g. with radius"""
        try:
            self.update_fit_info(
                'polnomial coefficients for {} : Rmse residuals {:.3f},\n Terms e.g. height = a + b*y + c*x + d*y^2 + e*xy...: {}\n'.format(
                    item, rmse(scan_fit[item]),
                    np.array2string(coef_scan[item], precision=2, separator=' , ', suppress_small=True)))
            if enDeg >= 1 and item == 'slopes_x':
                rc_tan = self.get_radius(item, coef_scan, scan, scale=1, i=1, stDeg=stDeg)
                self.update_param_info('{}: Tangential radius {} '.format(item, rc_tan))
                scan_fit['{}_Rc_tan'.format(item)] = rc_tan
            elif enDeg >= 1 and item == 'slopes_y':
                rc_sag = self.get_radius(item, coef_scan, scan, scale=1, i=1, stDeg=stDeg)
                self.update_param_info('{}: Sagittal radius {} '.format(item, rc_sag))
                scan_fit['{}_Rc_sag'.format(item)] = rc_sag
            elif enDeg >= 2 and item == 'height':
                if scan[item].ndim > 1:
                    rc_tan = self.get_radius(item, coef_scan, scan, scale=0.5, i=5, stDeg=stDeg)
                    rc_sag = self.get_radius(item, coef_scan, scan, scale=0.5, i=3, stDeg=stDeg)
                    self.update_param_info(
                        '{}: Tangential radius {}, \nSagittal radius {} '.format(item, rc_tan, rc_sag))
                    scan_fit['{}_Rc_tan'.format(item)] = rc_tan
                    scan_fit['{}_Rc_sag'.format(item)] = rc_sag
                else:
                    rc_tan = self.get_radius(item, coef_scan, scan, scale=0.5, i=2, stDeg=stDeg)
                    self.update_param_info('{}: Tangential radius {}'.format(item, rc_tan))
                    scan_fit['{}_Rc_tan'.format(item)] = rc_tan

        except Exception as e:
            print(e)
            self.Error.unknown(repr(e))

    def update_info_terms(self, item, scan_fit, coef_scan, scan):
        """Upate info box after filter terms is applied"""
        try:
            self.update_fit_info(
                'Terms fitted for {} : Rmse residuals {:.3f},\n Terms [piston, ty, tx, cyly, ast, cylx, sph] : {}\n'.format(
                    item, rmse(scan_fit[item]),
                    np.array2string(coef_scan[item], precision=2, separator=' , ', suppress_small=True)))
            scale = 0.5
            if self.filter_terms[3] and np.any(coef_scan[item][..., 3]):
                rc_sag = self.get_radius(item, coef_scan, scan, scale=scale, i=3)
                self.update_param_info('{}: Sagittal radius {} '.format(item, rc_sag))
                scan_fit['{}_Rc_sag'.format(item)] = rc_sag
            if self.filter_terms[5] and np.any(coef_scan[item][..., 5]):
                rc_tan = self.get_radius(item, coef_scan, scan, scale=scale, i=5)
                self.update_param_info('{}: Tangential radius {} '.format(item, rc_tan))
                scan_fit['{}_Rc_tan'.format(item)] = rc_tan
            if self.filter_terms[6] and np.any(coef_scan[item][..., 6]):
                rc = self.get_radius(item, coef_scan, scan, scale=scale, i=6)
                self.update_param_info('{}: Radius (spherical) {} '.format(item, rc))
                scan_fit['{}_radius'.format(item)] = rc
        except Exception as e:
            print(e)
            self.Error.unknown(repr(e))

    def get_radius(self, item, coef, scan, scale, i, stDeg=None):
        """Getradius from fit coefficients"""
        from astropy import units as u
        offset = 0
        if stDeg is None:
            stDeg = self.poly_deg_start
        if stDeg > 0:
            offset = nbTerms(stDeg - 1)
            if offset > i:
                offset = 0
        val = scale * 1 / coef[item][..., i - offset]
        try:
            if isinstance(scan[item], MetrologyData):
                # pix_size = scan[item].pix_size_detector
                pix_units = scan[item].get_axis_units_detector()
                if item == 'height':
                    val = val * ((pix_units[-1].to('m')) ** 2 / scan[item].unit.to('m')) * u.m
                elif item in ['slopes_x', 'slopes_y']:
                    val = val * (pix_units[-1].to('m') / scan[item].unit.to(
                        'rad')) * u.m  # .to('', equivalencies=u.dimensionless_angles())
                if np.all(val > 1000 * u.m):
                    val = val.to('km')
        except Exception as e:
            print(e)
        val = np.round(val, 5)
        return val

    @staticmethod
    def _format_coef(coef, first_only=False):
        if first_only and isinstance(coef, np.ndarray) and coef.ndim > 1:
            slc = tuple([slice(0)] * (coef.ndim - 1) + [slice(None)])
            coef_first = coef[slc]
            return np.array2string(coef_first, precision=7, separator=' , ', suppress_small=True)
        else:
            return np.array2string(coef, precision=7, separator=' , ', suppress_small=True)

    def filter(self):
        # Format C[0] + C[1]*Y + C[2]*X + C[3]*Y2+ C[4]*XY + C[5]*X2 + C[6]*(Y2+X2) for heights
        # Format C[2] + C[4]*Y + 2*C[5]*X for slopes x
        # Format C[1] + C[4]*X + 2*C[3]*Y for slopes y
        self.source = self.TERMS
        self.update_filter()

    def update_filter(self):
        self.filter_terms = [0, 0, 0, 0, 0, 0, 0]
        if self.piston:
            self.filter_terms[0] = 1
        if self.tiltY:
            self.filter_terms[1] = 1
        if self.tiltX:
            self.filter_terms[2] = 1
        if self.tilt:
            self.filter_terms[1] = 1
            self.filter_terms[2] = 1
        if self.cylY:
            self.filter_terms[3] = 1
        if self.cylX:
            self.filter_terms[5] = 1
        if self.sphere:
            self.filter_terms[6] = 1
        if self.ast:
            self.filter_terms[4] = 1

    def get_filter_terms(self):
        retArr = []
        if self.source == self.TERMS:
            if self.piston:
                retArr.append('Piston')
            if self.tilt:
                retArr.append('Tilt (XY)')
            if self.tiltX:
                retArr.append('Tilt X')
            if self.tiltY:
                retArr.append('Tilt Y')
            if self.sphere:
                retArr.append('Sphere')
            if self.cylX:
                retArr.append('Cylinder X')
            if self.cylY:
                retArr.append('Cylinder Y')
            if self.ast:
                retArr.append('Astigmatism')
        return retArr

    def sel_tilt_x(self):
        if self.tiltX:
            self.tilt = False

    def sel_tilt_y(self):
        if self.tiltY:
            self.tilt = False

    def sel_tilt(self):
        if self.tilt:
            self.tiltX = False
            self.tiltY = False

    def sel_cyl_x(self):
        if self.cylX:
            self.sphere = False

    def sel_cyl_y(self):
        if self.cylY:
            self.sphere = False

    def sel_sph(self):
        if self.sphere:
            self.cylX = False
            self.cylY = False

    def enable_options(self):
        self.controls.piston.setEnabled(True)
        self.controls.tilt.setEnabled(True)
        self.controls.tiltX.setEnabled(True)
        self.controls.tiltY.setEnabled(True)
        self.controls.sphere.setEnabled(True)
        self.controls.cylX.setEnabled(True)
        self.controls.cylY.setEnabled(True)
        self.controls.ast.setEnabled(True)

        self.controls.sel_poly.setEnabled(True)
        self.controls.poly_deg.setEnabled(True)
        self.controls.ellipse_p.setEnabled(True)
        self.controls.ellipse_q.setEnabled(True)
        self.controls.ellipse_theta.setEnabled(True)
        self.controls.ellipse_center_offset.setEnabled(True)

        # if not has_key_in_dict('height', self.data_in):
        #     self.controls.piston.setEnabled(False)
        #     self.controls.sphere.setEnabled(False)

    def disable_options(self):
        # setattr(self, 'source', self.DEFAULT)
        self.controls.piston.setEnabled(False)
        self.controls.tilt.setEnabled(False)
        self.controls.tiltX.setEnabled(False)
        self.controls.tiltY.setEnabled(False)
        self.controls.sphere.setEnabled(False)
        self.controls.cylX.setEnabled(False)
        self.controls.cylY.setEnabled(False)
        self.controls.ast.setEnabled(False)

        self.controls.sel_poly.setEnabled(False)
        self.controls.poly_deg.setEnabled(False)
        self.controls.ellipse_p.setEnabled(False)
        self.controls.ellipse_q.setEnabled(False)
        self.controls.ellipse_theta.setEnabled(False)
        self.controls.ellipse_center_offset.setEnabled(False)
