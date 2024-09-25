# coding=utf-8
"""
https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks
"""

import numpy as np
from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Output, Input
from PyQt5.QtCore import QSize
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QSizePolicy as Policy, QSizePolicy
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from PyLOSt.algorithms.util.util_math import nbTerms
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.util_functions import get_default_data_names, fit_nD_metrology

DEG_TO_MRAD = 17.4533


class OWFileFeatures(widget.OWWidget):
    name = 'Data Features'
    description = 'Loads data from file sequence and create features table.'
    icon = "../icons/features.svg"
    priority = 1000

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data_terms = Output('terms', Table)
        data_allpix = Output('all_pixels', Table)
        data_pix_seq = Output('pixel_sequence', Table)
        data_cnn = Output('cnn_images', dict, auto_summary=False)

    sel_terms_items = {'SELECT': 'Select', 'BASIC': 'Basic', 'POLY': 'Polynomial', 'LEGENDRE': 'Legendre polynomial',
                       'ZERNIKE': 'Zernike polynomial'}

    chk_terms = Setting(False, schema_only=True)
    chk_all_pix = Setting(False, schema_only=True)
    chk_pix_seq = Setting(False, schema_only=True)
    chk_conv = Setting(False, schema_only=True)

    sel_terms = Setting(sel_terms_items['SELECT'], schema_only=True)
    piston = Setting(False, schema_only=True)
    tilt = Setting(True, schema_only=True)
    tiltX = Setting(False, schema_only=True)
    tiltY = Setting(False, schema_only=True)
    sphere = Setting(False, schema_only=True)
    cylX = Setting(False, schema_only=True)
    cylY = Setting(False, schema_only=True)
    ast = Setting(False, schema_only=True)

    poly_deg = Setting(0, schema_only=True)
    poly_deg_start = Setting(0, schema_only=True)
    scale_data = Setting(1.0, schema_only=True)
    center_target = Setting(False, schema_only=True)

    normalize_inputs_pv = Setting(False, schema_only=True)
    normalize_inputs_zscore = Setting(False, schema_only=True)
    normalize_inputs_tanh = Setting(False, schema_only=True)

    want_main_area = False

    class Information(widget.OWWidget.Information):
        info = Msg("Info:\n{}")

    class Error(widget.OWWidget.Error):
        unknown = Msg("Read error:\n{}")

    def __init__(self):
        super().__init__()
        self.data_out = {}
        self.data_in = {}
        self.source = 0
        self.filename_seq = []
        self.DEFAULT_DATA_NAMES = get_default_data_names()
        self.DATA_NAMES = []
        self.default_axis_names = ['Motor', 'Y', 'X']
        self.default_dim_detector = [-2, -1]
        self.filter_terms = [0, 0, 0, 0, 0, 0, 0]

        box = gui.hBox(self.controlArea, "Info", stretch=1)
        self.infolabel = gui.widgetLabel(box, 'No data loaded.', stretch=9)
        applyBtn = gui.button(box, self, "Apply", callback=self.apply_changes, autoDefault=False, stretch=1)
        applyBtn.setSizePolicy(Policy.Fixed, Policy.Fixed)

        le0 = gui.lineEdit(self.controlArea, self, 'scale_data', 'Scale data', orientation=Qt.Horizontal,
                           sizePolicy=(QSizePolicy.Fixed, QSizePolicy.Fixed))
        gui.checkBox(self.controlArea, self, 'center_target', 'Center target/output',
                     sizePolicy=(QSizePolicy.Fixed, QSizePolicy.Fixed))
        layout = QGridLayout()
        gui.widgetBox(self.controlArea, "Options", margin=10, orientation=layout, addSpace=True, stretch=8)
        lbl00 = gui.widgetLabel(None, 'Outputs :')
        ch01 = gui.checkBox(None, self, "chk_terms", "Terms (Features)")
        ch02 = gui.checkBox(None, self, "chk_all_pix", "All pixels")
        ch03 = gui.checkBox(None, self, "chk_pix_seq", "Pixel sequence")
        ch04 = gui.checkBox(None, self, "chk_conv", "Images for CNN")
        layout.addWidget(lbl00, 0, 0, Qt.AlignVCenter)
        layout.addWidget(ch01, 0, 1, Qt.AlignVCenter)
        layout.addWidget(ch02, 0, 2, Qt.AlignVCenter)
        layout.addWidget(ch03, 0, 3, Qt.AlignVCenter)
        layout.addWidget(ch04, 0, 4, Qt.AlignVCenter)

        lbl10 = gui.widgetLabel(None, 'Terms :')
        self.combo_terms = gui.comboBox(box, self, "sel_terms", sendSelectedValue=True,
                                        items=list(self.sel_terms_items.values()),
                                        sizePolicy=Policy(Policy.Fixed, Policy.Fixed),
                                        callback=self.combo_terms_selection)
        layout.addWidget(lbl10, 1, 0)
        layout.addWidget(self.combo_terms, 1, 1)

        self.lbl_basic = gui.widgetLabel(None, 'Terms (basic) :')
        ch21 = gui.checkBox(None, self, "piston", "Piston", callback=self.filter)
        ch31 = gui.checkBox(None, self, "tilt", "Tilt (XY)", callback=[self.sel_tilt, self.filter])
        ch32 = gui.checkBox(None, self, "tiltX", "Tilt X", callback=[self.sel_tilt_x, self.filter])
        ch33 = gui.checkBox(None, self, "tiltY", "Tilt Y", callback=[self.sel_tilt_y, self.filter])
        ch41 = gui.checkBox(None, self, "sphere", "Sphere", callback=[self.sel_sph, self.filter])
        ch42 = gui.checkBox(None, self, "cylX", "Cylinder X", callback=[self.sel_cyl_x, self.filter])
        ch43 = gui.checkBox(None, self, "cylY", "Cylinder Y", callback=[self.sel_cyl_y, self.filter])
        ch51 = gui.checkBox(None, self, "ast", "Astigmatism", callback=self.filter)
        layout.addWidget(self.lbl_basic, 2, 0, Qt.AlignVCenter)
        layout.addWidget(ch21, 2, 1, Qt.AlignVCenter)
        layout.addWidget(ch31, 3, 1, Qt.AlignVCenter)
        layout.addWidget(ch32, 3, 2, Qt.AlignVCenter)
        layout.addWidget(ch33, 3, 3, Qt.AlignVCenter)
        layout.addWidget(ch41, 4, 1, Qt.AlignVCenter)
        layout.addWidget(ch42, 4, 2, Qt.AlignVCenter)
        layout.addWidget(ch43, 4, 3, Qt.AlignVCenter)
        layout.addWidget(ch51, 5, 1, Qt.AlignVCenter)

        self.lbl_poly = gui.widgetLabel(None, 'Terms (polynomial) :')
        box = gui.hBox(None)
        self.polyDegree = gui.lineEdit(box, self, 'poly_deg', "Degree:", labelWidth=50, orientation=Qt.Horizontal,
                                       sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        self.polyDegStart = gui.lineEdit(box, self, 'poly_deg_start', "Start degree:", labelWidth=75,
                                         orientation=Qt.Horizontal, sizePolicy=Policy(Policy.Fixed, Policy.Fixed))
        layout.addWidget(self.lbl_poly, 6, 0)
        layout.addWidget(box, 6, 1, 1, 3)

        ch71 = gui.checkBox(None, self, "normalize_inputs_pv", "Normalize inputs pv")
        ch81 = gui.checkBox(None, self, "normalize_inputs_zscore", "Normalize inputs z-score (standardize)")
        ch91 = gui.checkBox(None, self, "normalize_inputs_tanh", "Normalize inputs tanh estimators")
        layout.addWidget(ch71, 7, 1, 1, 3)
        layout.addWidget(ch81, 8, 1, 1, 3)
        layout.addWidget(ch91, 9, 1, 1, 3)

    def sizeHint(self):
        return QSize(500, 50)

    @Inputs.data
    def set_data(self, data):
        if data is not None:
            self.data_in = data
            self.load_data()
        else:
            self.data_in = {}
            self.infolabel.setText('No data')

    def filter(self):
        # Format C[0] + C[1]*Y + C[2]*X + C[3]*Y2+ C[4]*XY + C[5]*X2 + C[6]*(Y2+X2) for heights
        # Format C[2] + C[4]*Y + 2*C[5]*X for slopes x
        # Format C[1] + C[4]*X + 2*C[3]*Y for slopes y
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

    def hide_terms_basic(self):
        self.lbl_basic.hide()
        self.controls.piston.hide()
        self.controls.tilt.hide()
        self.controls.tiltX.hide()
        self.controls.tiltY.hide()
        self.controls.sphere.hide()
        self.controls.cylX.hide()
        self.controls.cylY.hide()
        self.controls.ast.hide()

    def show_terms_basic(self):
        self.lbl_basic.show()
        self.controls.piston.show()
        self.controls.tilt.show()
        self.controls.tiltX.show()
        self.controls.tiltY.show()
        self.controls.sphere.show()
        self.controls.cylX.show()
        self.controls.cylY.show()
        self.controls.ast.show()

    def show_terms_poly(self):
        self.lbl_poly.show()
        self.polyDegree.parent().show()
        self.polyDegStart.parent().show()

    def hide_terms_poly(self):
        self.lbl_poly.hide()
        self.polyDegree.parent().hide()
        self.polyDegStart.parent().hide()

    def hide_all_terms(self):
        self.hide_terms_basic()
        self.hide_terms_poly()

    def combo_terms_selection(self):
        if self.sel_terms == self.sel_terms_items['SELECT']:
            self.hide_all_terms()
        elif self.sel_terms == self.sel_terms_items['BASIC']:
            self.show_terms_basic()
            self.hide_terms_poly()
        else:
            self.hide_terms_basic()
            self.show_terms_poly()

    @staticmethod
    def join_arr(a, b):
        if b is None:
            return a
        elif a is None:
            return b
        else:
            if a.ndim == 1:
                a = a[:, np.newaxis]
            if b.ndim == 1:
                b = b[:, np.newaxis]
            return np.concatenate([a, b], axis=1)

    def load_data(self):
        # self.apply_changes()
        self.combo_terms_selection()

    def apply_changes(self):
        if self.chk_all_pix:
            self.load_all_pix()
        if self.chk_terms:
            self.load_terms()
        if self.chk_pix_seq:
            self.load_pix_seq()
        if self.chk_conv:
            self.load_cnn_images()

    def load_all_pix(self):
        names_features = []
        data_features = None
        names_targets = []
        data_targets = None
        motors = None
        prefixes = {'height': 'H', 'slopes_x': 'SX', 'slopes_y': 'SY'}
        for key in self.DEFAULT_DATA_NAMES:
            if key in self.data_in:
                arr = self.scale_data * self.data_in[key].reshape(self.data_in[key].shape[0], -1)
                col_msk = np.all(np.isfinite(arr), axis=0)
                arr = arr[:, col_msk]
                names_features += [prefixes[key] + '{}'.format(x) for x in col_msk.nonzero()[0]]
                names_features += [prefixes[key] + '_mean']
                if isinstance(arr, MetrologyData):
                    if np.any(arr.motors):
                        motors = arr.motors
                    arr = arr.value
                data_features = self.join_arr(data_features, arr)
                arr_m = np.nanmean(np.nanmean(self.data_in[key], axis=-1), axis=-1)
                if isinstance(arr_m, MetrologyData):
                    arr_m = arr_m.value
                data_features = self.join_arr(data_features, arr_m)

        if motors is not None:
            for m in motors:
                arr = np.array(m['values']).reshape(-1, 1)
                names_targets += [m['name']]
                data_targets = self.join_arr(data_targets, arr)
                # TODO
                if m['name'] == 'motor_RY' and 'slopes_x' in self.data_in:
                    names_targets += ['sx_error']
                    arr = np.nanmean(np.nanmean(self.data_in['slopes_x'], axis=-1), axis=-1).value + np.array(
                        m['values'])
                    data_targets = self.join_arr(data_targets, arr)

        else:
            for key in ['motorX', 'motorY', 'motorZ', 'motorRx', 'motorRy', 'motorRz']:
                if key in self.data_in:
                    arr = self.data_in[key].reshape(-1, 1)
                    names_targets += [key]
                    data_targets = self.join_arr(data_targets, arr)

        self.send_output('allpix', names_features, data_features, names_targets, data_targets)

    def load_terms(self):
        names_features = []
        data_features = None
        names_targets = []
        data_targets = None
        motors = None
        prefixes = {'height': 'H', 'slopes_x': 'SX', 'slopes_y': 'SY'}
        for key in self.DEFAULT_DATA_NAMES:
            if key in self.data_in:
                if self.sel_terms == self.sel_terms_items['BASIC']:
                    coef, _, _ = fit_nD_metrology(self.scale_data * self.data_in[key],
                                                  pix_size=self.data_in[key].pix_size, retResd=False,
                                                  filter_terms_poly=self.filter_terms, dtyp=key)
                    if self.piston and key in ['height']:
                        names_features += ['piston']
                        data_features = self.join_arr(data_features, coef[:, 0])
                    if (self.tilt or self.tiltX) and key in ['height', 'slopes_x']:
                        names_features += ['tilt_x']
                        data_features = self.join_arr(data_features, coef[:, 2])
                    if (self.tilt or self.tiltY) and key in ['height', 'slopes_y']:
                        names_features += ['tilt_y']
                        data_features = self.join_arr(data_features, coef[:, 1])
                    if self.cylX and key in ['height', 'slopes_x']:
                        names_features += ['cylinder_x']
                        data_features = self.join_arr(data_features, coef[:, 5])
                    if self.cylY and key in ['height', 'slopes_y']:
                        names_features += ['cylinder_y']
                        data_features = self.join_arr(data_features, coef[:, 3])
                    if self.sphere and key in ['height']:
                        names_features += ['sphere']
                        data_features = self.join_arr(data_features, coef[:, 6])
                    if self.ast and key in ['height', 'slopes_x', 'slopes_y']:
                        ast_dict = {'height': 'astigmatism', 'slopes_x': 'astigmatism_x', 'slopes_y': 'astigmatism_y'}
                        names_features += [ast_dict[key]]
                        data_features = self.join_arr(data_features, coef[:, 4])
                elif self.sel_terms in [self.sel_terms_items['POLY'], self.sel_terms_items['LEGENDRE'],
                                        self.sel_terms_items['ZERNIKE']]:
                    typ_dict = {self.sel_terms_items['POLY']: 'poly',
                                self.sel_terms_items['LEGENDRE']: 'legendre',
                                self.sel_terms_items['ZERNIKE']: 'zernike'}
                    typ = typ_dict[self.sel_terms]
                    offset = nbTerms(self.poly_deg_start - 1)
                    coef, _, _ = fit_nD_metrology(self.data_in[key], pix_size=self.data_in[key].pix_size,
                                                  degree=self.poly_deg,
                                                  start_degree=self.poly_deg_start, dtyp=key, typ=typ)
                    names_features += [key + '_' + typ + '_{}'.format(x + offset) for x in np.arange(coef.shape[1])]
                    data_features = self.join_arr(data_features, coef)

                if isinstance(self.data_in[key], MetrologyData):
                    if np.any(self.data_in[key].motors):
                        motors = self.data_in[key].motors
                if isinstance(data_features, MetrologyData):
                    data_features = data_features.value
        if motors is not None:
            for m in motors:
                arr = np.array(m['values']).reshape(-1, 1)
                names_targets += [m['name']]
                data_targets = self.join_arr(data_targets, arr)
                # TODO
                if m['name'] == 'motor_RY' and 'slopes_x' in self.data_in:
                    names_targets += ['sx_error']
                    arr = np.nanmean(np.nanmean(self.data_in['slopes_x'], axis=-1), axis=-1).value + np.array(
                        m['values'])
                    data_targets = self.join_arr(data_targets, arr)
        else:
            for key in ['motorX', 'motorY', 'motorZ', 'motorRx', 'motorRy', 'motorRz']:
                if key in self.data_in:
                    arr = self.data_in[key].reshape(-1, 1)
                    names_targets += [key]
                    data_targets = self.join_arr(data_targets, arr)
        self.send_output('terms', names_features, data_features, names_targets, data_targets)

    def load_pix_seq(self):
        names_features = []
        data_features = None
        names_targets = []
        data_targets = None
        dy = None
        motors = None
        if 'height' in self.data_in:
            d = self.scale_data * self.data_in['height']
            shp = d.shape
            msk = np.isfinite(d)
            f = ['height']
        elif 'slopes_x' in self.data_in:
            d = self.scale_data * self.data_in['slopes_x']
            shp = d.shape
            dy = self.scale_data * self.data_in['slopes_y'] if 'slopes_y' in self.data_in else None
            msk = np.isfinite(d) * np.isfinite(dy) if dy is not None else np.isfinite(d)
            f = ['slopes_x']
            if dy is not None:
                f += ['slopes_y']

        if isinstance(d, MetrologyData):
            if np.any(d.motors):
                motors = d.motors
        if isinstance(d, MetrologyData):
            d = d.value
        if dy is not None and isinstance(dy, MetrologyData):
            dy = dy.value

        x = np.arange(d.shape[-1])
        y = np.arange(d.shape[-2])
        xx, yy = np.meshgrid(x, y)
        xx = np.tile(xx, (d.shape[0], 1, 1))
        yy = np.tile(yy, (d.shape[0], 1, 1))
        names_features += ['x', 'y']
        data_features = self.join_arr(data_features, xx[msk])
        data_features = self.join_arr(data_features, yy[msk])
        names_features += f
        data_features = self.join_arr(data_features, d[msk])
        if dy is not None:
            data_features = self.join_arr(data_features, dy[msk])

        if motors is not None:
            for m in motors:
                arr = np.array(m['values']).reshape(-1, 1, 1)
                names_targets += [m['name']]
                arr = np.tile(arr, (1, shp[1], shp[2]))
                data_targets = self.join_arr(data_targets, arr[msk])
        else:
            for key in ['motorX', 'motorY', 'motorZ', 'motorRx', 'motorRy', 'motorRz']:
                if key in self.data_in:
                    arr = self.data_in[key].reshape(-1, 1)
                    names_targets += [key]
                    arr = np.tile(arr, (1, d.shape[1], d.shape[2]))
                    data_targets = self.join_arr(data_targets, arr[msk])

        self.send_output('seq', names_features, data_features, names_targets, data_targets)

    def load_cnn_images(self):
        names_features = []
        data_features = None
        names_targets = []
        data_targets = None
        dy = None
        motors = None
        if 'height' in self.data_in:
            d = self.scale_data * self.data_in['height']
            shp = d.shape
            msk = np.isfinite(d)
            data = d[:, np.newaxis, :, :]
            f = ['height']
        elif 'slopes_x' in self.data_in:
            d = self.scale_data * self.data_in['slopes_x']
            shp = d.shape
            dy = self.scale_data * self.data_in['slopes_y'] if 'slopes_y' in self.data_in else None
            msk = np.isfinite(d) * np.isfinite(dy) if dy is not None else np.isfinite(d)
            data = d[:, np.newaxis, :, :]
            f = ['slopes_x']
            if dy is not None:
                f += ['slopes_y']
                data = np.concatenate([data, d[:, np.newaxis, :, :]], axis=1)

        if isinstance(d, MetrologyData):
            if np.any(d.motors):
                motors = d.motors
        if isinstance(data, MetrologyData):
            data = data.value

        if motors is not None:
            for m in motors:
                arr = np.array(m['values']).reshape(-1, 1)
                if m['name'] in ['motor_RX', 'motor_RY']:
                    names_targets += [m['name']]
                    data_targets = self.join_arr(data_targets, arr)
        else:
            for key in ['motorRx', 'motorRy']:  # ['motorX', 'motorY', 'motorZ', 'motorRx', 'motorRy', 'motorRz']:
                if key in self.data_in:
                    arr = self.data_in[key].reshape(-1, 1)
                    names_targets += [key]
                    data_targets = self.join_arr(data_targets, arr)
        data = np.nan_to_num(data, nan=0.0)
        out = {'X': data, 'Y': data_targets}
        self.Outputs.data_cnn.send(out)

    def send_output(self, typ, names_features, data_features, names_targets, data_targets):
        if len(names_features) > 0:
            if self.normalize_inputs_pv:
                data_features = (data_features - np.nanmean(data_features, axis=0)) / (
                        np.nanmax(data_features, axis=0) - np.nanmin(data_features, axis=0))
            elif self.normalize_inputs_zscore:
                data_features = (data_features - np.nanmean(data_features, axis=0)) / np.nanstd(data_features, axis=0)
            elif self.normalize_inputs_tanh:
                # https://www.cs.ccu.edu.tw/~wylin/BA/Fusion_of_Biometrics_II.ppt
                data_features = 0.5 + 0.5 * np.tanh(
                    0.01 * (data_features - np.nanmean(data_features, axis=0)) / np.nanstd(data_features, axis=0))
            domain = Domain(attributes=[ContinuousVariable.make('{}'.format(x)) for x in names_features],
                            class_vars=[ContinuousVariable.make('{}'.format(x)) for x in names_targets])
            features = Table.from_numpy(domain, data_features, data_targets)
            if typ == 'terms':
                self.Outputs.data_terms.send(features)
            elif typ == 'allpix':
                self.Outputs.data_allpix.send(features)
            elif typ == 'seq':
                self.Outputs.data_pix_seq.send(features)
