# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.widget import Msg
from sympy import lambdify, parse_expr, symbols

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.math import pv
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets

from AnyQt.QtWidgets import QFileDialog
from orangewidget.utils.filedialogs import RecentPathsWComboMixin, format_filter
from Orange.data import FileFormat
from pylost_widgets.util.ow_filereaders import *
from pathlib import Path

class OWSimulateMirror(PylostWidgets, PylostBase, RecentPathsWComboMixin):
    name = 'Simulate Mirror'
    description = 'Simulate mirror with different shapes and create subapertures'
    icon = "../icons/mirror.svg"
    priority = 91

    class Outputs:
        data_subap = Output('subapertures', dict, auto_summary=False, default=True)
        data_mir = Output('mirror', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)

    FLAT, SPH, CYL, ELLIPSE, OTHER = range(5)
    MIRROR_SHAPES = ['Flat', 'Spherical', 'Cylindrical', 'Elliptical (tangential)', 'Other']
    mir_shape = Setting(FLAT, schema_only=True)
    mir_length = Setting(0.0, schema_only=True)
    mir_width = Setting(0.0, schema_only=True)
    mir_pix_x = Setting(0.0, schema_only=True)
    mir_pix_y = Setting(0.0, schema_only=True)
    rand_std = Setting(0.0, schema_only=True)

    # Sphere params
    radius = Setting(0.0, schema_only=True)
    # Cylinders
    radius_tan = Setting(0.0, schema_only=True)
    radius_sag = Setting(0.0, schema_only=True)
    # Ellipse
    p = Setting(0.0, schema_only=True)
    q = Setting(0.0, schema_only=True)
    theta = Setting(0.0, schema_only=True)
    # Other
    eqn_shape = Setting('', schema_only=True)

    # Shape errors
    pv_shape_err = Setting(1.0, schema_only=True)
    FE_RANDOM, FE_OTHER = range(2)
    FIGURE_ERRORS = ['Generate randomly', 'Enter equation']
    mir_shape_err = Setting(FE_RANDOM, schema_only=True)
    eqn_fe = Setting('', schema_only=True)
    noise_fe = Setting(0.0, schema_only=True)

    # Subapertures
    gen_subaps = Setting(True, schema_only=True)
    count_subaps = Setting(1, schema_only=True)
    subap_length = Setting(0.0, schema_only=True)
    subap_width = Setting(0.0, schema_only=True)
    step_x = Setting(0.0, schema_only=True)
    step_y = Setting(0.0, schema_only=True)
    step_x_var = Setting(0.0, schema_only=True)
    step_y_var = Setting(0.0, schema_only=True)

    # Refernce errors
    REF_NONE, REF_FILE, REF_OTHER = range(3)
    pv_ref_err = Setting(1.0, schema_only=True)
    REF_ERRORS = ['None', 'Load from file', 'Enter equation']
    ref_err = Setting(REF_NONE, schema_only=True)
    eqn_ref = Setting('', schema_only=True)
    noise_ref = Setting(0.0, schema_only=True)
    ref_path = Setting('', schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        self._clean_recentpaths()
        PylostBase.__init__(self)
        RecentPathsWComboMixin.__init__(self)

        box = super().init_info(module=True)
        gui.button(box, self, 'Generate random parameters', callback=self.gen_random_params, autoDefault=False,
                   stretch=1, sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.btnApply = gui.button(box, self, 'Simulate mirror', callback=self.simulateData, autoDefault=False,
                                   stretch=1, sizePolicy=(Policy.Fixed, Policy.Fixed))

        mbox = gui.hBox(self.controlArea)
        left_box = gui.vBox(mbox)
        box = gui.vBox(left_box, "Mirror parameters")
        hbox = gui.hBox(box, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(hbox, self, 'mir_length', 'Mirror length (mm)', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(hbox, self, 'mir_width', 'Mirror width (mm)', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        hbox = gui.hBox(box, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(hbox, self, 'mir_pix_x', 'Pixel size X (mm)', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.change_pix_x)
        gui.lineEdit(hbox, self, 'mir_pix_y', 'Pixel size Y (mm)', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.change_pix_y)

        gui.comboBox(box, self, 'mir_shape', label='Mirror shape', orientation=Qt.Horizontal, items=self.MIRROR_SHAPES,
                     callback=self.change_shape, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'radius', 'Radius (m)', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'radius_tan', 'Radius tangential (m)', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'radius_sag', 'Radius sagittal (m)', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'p', 'P (m)', orientation=Qt.Horizontal, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'q', 'Q (m)', orientation=Qt.Horizontal, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'theta', 'Theta (mrad)', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'eqn_shape', 'Equation(x,y) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), controlWidth=128)
        self.label_eqn_shape = gui.label(box, self, '')
        self.change_shape()

        box = gui.vBox(left_box, "Mirror figure errors")
        gui.lineEdit(box, self, 'pv_shape_err', 'Peak to valley (nm) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.comboBox(box, self, 'mir_shape_err', label='Mirror shape errors', orientation=Qt.Horizontal,
                     items=self.FIGURE_ERRORS, callback=self.change_shape_err, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'eqn_fe', 'Equation(x,y) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), controlWidth=128)
        gui.lineEdit(box, self, 'noise_fe', 'percentage of noise added :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.label_eqn_shape_err = gui.label(box, self, '')
        self.change_shape_err()

        right_box = gui.vBox(mbox)
        box = gui.vBox(right_box, "Subapertures")
        gui.checkBox(box, self, 'gen_subaps', 'Generate subapertures', sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'rand_std', 'Random errors std (nm)', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'count_subaps', 'Number of subapertures :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'subap_length', 'Length of subaperture (mm) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(box, self, 'subap_width', 'Width of subaperture (mm) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        hbox = gui.hBox(box, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(hbox, self, 'step_x', 'Step X (mm) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.update_motor_x)
        gui.lineEdit(hbox, self, 'step_x_var', ' with random variation in +- (mm) ', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.update_motor_x)
        hbox = gui.hBox(box, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(hbox, self, 'step_y', 'Step Y (mm) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.update_motor_y)
        gui.lineEdit(hbox, self, 'step_y_var', ' with random variation in +- (mm) ', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.update_motor_y)

        rbox = gui.vBox(right_box, "Reference errors")
        gui.lineEdit(rbox, self, 'pv_ref_err', 'Peak to valley (nm) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.comboBox(rbox, self, 'ref_err', label='Reference errors', orientation=Qt.Horizontal, items=self.REF_ERRORS,
                     callback=self.change_ref_err, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(rbox, self, 'eqn_ref', 'Equation(x,y) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), controlWidth=128)
        gui.lineEdit(rbox, self, 'noise_ref', 'percentage of noise added :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.change_ref_err()

        self.mir_func = None
        self.mx = np.array([0.0])
        self.my = np.array([0.0])

        self.readers = FileFormat.readers
        self.ref_loaded = None

    def sizeHint(self):
        return QSize(400, 50)

    def _clean_recentpaths(self):
        pathlist = []
        for i, item in enumerate(self.recent_paths):
            if i > 20:
                break
            if Path(item.abspath).exists():
                pathlist.append(item)
        self.recent_paths = pathlist

    @property
    def last_folder(self):
        folder = self.last_path()
        if folder is None:
            return Path.home()
        folder = Path(folder)
        if folder.is_file():
            folder = folder.parent
        return folder

    def open_files_dialog(self, start_dir=None, title="Open..."):
        return
        filters = []
        for reader in self.readers.values():
            filter_str = format_filter(reader)
            if filter_str not in filters:
                filters.append(filter_str)
        if start_dir is None:
            start_dir = str(self.last_folder)
        self.ref_path, _ = QFileDialog.getOpenFileName(None, title, start_dir, ';;'.join(filters))
        if not self.ref_path:
            return None
        self.add_path(str(Path(self.ref_path).parent))
        self.ref_loaded = self._try_load(self.ref_path)
        return self.filename

    def _try_load(self, path):
        try:
            reader = FileFormat.get_reader(path)
            data = reader.read()
            if isinstance(data, dict):
                data = data.get('height', None)
            return data.si.value
        except Exception as ex:
            log.exception(ex)
            self.setStatusMessage('')
            return lambda x=ex: self.Error.unknown(str(x))

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Simulated mirror')

    def change_pix_x(self):
        if self.mir_pix_y == 0:
            self.mir_pix_y = self.mir_pix_x

    def change_pix_y(self):
        if self.mir_pix_x == 0:
            self.mir_pix_x = self.mir_pix_y

    def change_ref_err(self):
        self.controls.eqn_ref.parent().hide()
        self.controls.noise_ref.parent().hide()
        if self.ref_err == self.REF_OTHER:
            self.controls.eqn_ref.parent().show()
            self.controls.noise_ref.parent().show()
        if self.ref_err == self.REF_FILE:
            self.open_files_dialog()

    def change_shape_err(self):
        self.hide_shape_err_items()
        if self.mir_shape_err == self.FE_OTHER:
            self.controls.eqn_fe.parent().show()
            self.controls.noise_fe.parent().show()

    def hide_shape_err_items(self):
        self.controls.eqn_fe.parent().hide()
        self.controls.noise_fe.parent().hide()

    def change_shape(self):
        self.hide_params()
        if self.mir_shape == self.SPH:
            self.controls.radius.parent().show()
        if self.mir_shape == self.CYL:
            self.controls.radius_tan.parent().show()
            self.controls.radius_sag.parent().show()
        if self.mir_shape == self.ELLIPSE:
            self.controls.p.parent().show()
            self.controls.q.parent().show()
            self.controls.theta.parent().show()
        if self.mir_shape == self.OTHER:
            self.controls.eqn_shape.parent().show()

    def hide_params(self):
        self.controls.radius.parent().hide()
        self.controls.radius_tan.parent().hide()
        self.controls.radius_sag.parent().hide()
        self.controls.p.parent().hide()
        self.controls.q.parent().hide()
        self.controls.theta.parent().hide()
        self.controls.eqn_shape.parent().hide()

    @staticmethod
    def get_eqn(eqn):
        if eqn != '':
            x, y = symbols('x y')
            equation = parse_expr(eqn)#, transformations='all')
            return lambdify([x, y], equation)
        else:
            return None

    def gen_random_params(self):
        self.mir_length = np.round(10 ** np.random.uniform(1, 3), 4)
        self.mir_width = np.round(np.random.uniform(5, 50), 4)
        self.mir_pix_x = np.round(10 ** np.random.uniform(-2, 0), 4)
        self.mir_pix_y = self.mir_pix_x  # np.round(self.mir_pix_x * np.random.uniform(0.9, 1.1), 4)
        self.radius = np.round(10 ** np.random.uniform(0, 4), 4)
        self.radius_tan = np.round(10 ** np.random.uniform(0, 4), 4)
        self.radius_sag = np.round(10 ** np.random.uniform(0, 4), 4)
        self.pv_shape_err = np.round(10 ** np.random.uniform(-1, 1.2), 4)
        self.rand_std = np.round(10 ** np.random.uniform(-2, 1), 4)

        self.subap_width = np.round(self.mir_width, 4)
        self.subap_length = np.round(np.min([self.mir_length / 2, 150.0]), 4)
        self.step_y = 0.0
        self.step_x = np.round(self.subap_length * np.random.uniform(0.05, 0.4), 4)
        self.update_motor_x()
        self.update_motor_y()

    def update_count(self):
        if self.step_x > 0:
            self.count_subaps = int(1 + (self.mir_length - self.subap_length) / self.step_x)
        if self.step_y > 0:
            cy = int(1 + (self.mir_width - self.subap_width) / self.step_y)
            self.count_subaps = np.min(self.count_subaps, cy)

    def update_motor_x(self, update_count=True):
        if self.count_subaps <= 1 or update_count:
            self.update_count()
        if self.step_x > 0:
            self.mx = np.arange(self.count_subaps) * self.step_x
        if self.step_x_var > 0:
            self.mx += np.random.uniform(-1 * self.step_x_var, self.step_x_var, self.count_subaps)

        if pv(self.mx) + self.subap_length > self.mir_length:
            if self.count_subaps > 2:
                self.count_subaps -= 1
                self.update_motor_x(update_count=False)
            else:
                raise Exception('Unable to create subapertures. please recheck parameters')

    def update_motor_y(self, update_count=True):
        if self.count_subaps <= 1 or update_count:
            self.update_count()
        if self.step_y > 0:
            self.my = np.arange(self.count_subaps) * self.step_y
        if self.step_y_var > 0:
            self.my += np.random.uniform(-1 * self.step_y_var, self.step_y_var, self.count_subaps)

        if pv(self.my) + self.subap_width > self.mir_width:
            if self.count_subaps > 2:
                self.count_subaps -= 1
                self.update_motor_y(update_count=False)
            else:
                raise Exception('Unable to create subapertures. please recheck parameters')

    def simulateData(self):
        self.data_out = {}
        self.clear_messages()
        try:
            if self.mir_pix_x > 0 and self.mir_pix_y > 0:
                nx = int(self.mir_length / self.mir_pix_x)
                ny = int(self.mir_width / self.mir_pix_y)
                x = np.arange(nx).reshape(1, -1) * self.mir_pix_x  # in mm
                y = np.arange(ny).reshape(-1, 1) * self.mir_pix_y  # in mm
                x = x - np.nanmean(x)
                y = y - np.nanmean(y)
                x = x * 1e-3  # meters
                y = y * 1e-3  # meters
                height = self.get_shape(nx, ny, x, y)
                fig_err = self.get_figure_errors(nx, ny, x, y)
                height += fig_err * self.pv_shape_err / pv(fig_err)
                height_mir = MetrologyData(height, unit='nm', pix_size=[self.mir_pix_y, self.mir_pix_x],
                                           pix_unit='mm', dim_detector=[-2, -1], axis_names=['Y', 'X'])
                self.Outputs.data_mir.send({'height': height_mir})

                if self.count_subaps > 1 and self.gen_subaps:
                    height = self.split_subaps(height)
                    if self.rand_std > 0:
                        height += np.random.normal(scale=self.rand_std, size=height.shape)
                    motors = []
                    if np.any(self.mx):
                        motors += [{'name': 'motor_X', 'values': self.mx, 'axis': [-3], 'unit': 'mm'}]
                    if np.any(self.my):
                        motors += [{'name': 'motor_Y', 'values': self.my, 'axis': [-3], 'unit': 'mm'}]
                    height = MetrologyData(height, unit='nm', pix_size=[self.mir_pix_y, self.mir_pix_x],
                                           pix_unit='mm', dim_detector=[-2, -1], axis_names=['Motor', 'Y', 'X'],
                                           motors=motors)
                else:
                    height = MetrologyData(height, unit='nm', pix_size=[self.mir_pix_y, self.mir_pix_x],
                                           pix_unit='mm', dim_detector=[-2, -1], axis_names=['Y', 'X'])
                self.data_out['height'] = height
                self.update_comment('')
                self.setStatusMessage('Mirror generated successfully')
                self.info.set_output_summary('Mirror generated successfully')
            else:
                raise Exception('Please enter pixel size x and y')
        except Exception as e:
            self.Error.unknown(str(e))
        self.Outputs.data_subap.send(self.data_out)

    def split_subaps(self, height):
        self.mx_pix = (self.mx / self.mir_pix_x).astype(int) if np.any(self.mx) else np.zeros((self.count_subaps,),
                                                                                              dtype=int)
        self.my_pix = (self.my / self.mir_pix_y).astype(int) if np.any(self.my) else np.zeros((self.count_subaps,),
                                                                                              dtype=int)
        slen = int(self.subap_length / self.mir_pix_x)
        swid = int(self.subap_width / self.mir_pix_y)
        h = np.full((self.count_subaps, swid, slen), np.nan, dtype=float)
        if self.ref_err == self.REF_OTHER:
            func = self.get_eqn(self.eqn_ref)
            if func is None:
                raise Exception('Please enter an equation for reference errors')
            x = np.linspace(0, self.subap_length*1e-3, num=slen, endpoint=False)
            y = np.linspace(0, self.subap_width*1e-3, num=swid, endpoint=False)
            self.ref_loaded = func(x, y)
        for i in range(self.count_subaps):
            ox = self.mx_pix[i]
            oy = self.my_pix[i]
            h[i] = height[oy:oy + swid, ox:ox + slen]
            if self.ref_err == self.REF_OTHER or self.ref_err == self.REF_FILE:
                h[i] += self.ref_loaded * 1e9
                if self.noise_ref > 0.0:
                    h[i] += np.random.normal(scale=np.sqrt(np.max(h) * self.noise_ref), size=h.shape)
        return h

    def get_figure_errors(self, nx, ny, x, y):
        if self.mir_shape_err == self.FE_RANDOM:
            return np.random.random((ny, nx))
        elif self.mir_shape_err == self.FE_OTHER:
            func = self.get_eqn(self.eqn_fe)
            if func is None:
                raise Exception('Please enter an equation for figure errors')
            h = func(x, y)
            if self.noise_fe > 0.0:
                h += np.random.normal(scale=np.sqrt(np.max(h) * self.noise_fe), size=h.shape)
            return h * 1e9

    def get_shape(self, nx, ny, x, y):
        if self.mir_shape == self.FLAT:
            return np.full((ny, nx), 0.0, dtype=float)
        else:
            if self.mir_shape == self.SPH:
                if self.radius == 0:
                    raise Exception('Please enter radius for the sphere')
                R = self.radius  # in meters
                r = np.sqrt(x ** 2 + y ** 2)
                h = R - np.sqrt(R ** 2 - r ** 2)  # if r<<R, h = r**2/2R
                h = h - np.min(h)
                return h * 1e9  # in nanometers
            elif self.mir_shape == self.CYL:
                if self.radius_tan == 0 and self.radius_sag == 0:
                    raise Exception('Please enter either tangential or sagittal radius')
                ctan = 1 / self.radius_tan if self.radius_tan > 0 else 0.0
                csag = 1 / self.radius_sag if self.radius_sag > 0 else 0.0
                h = 0.5 * ctan * x ** 2 + 0.5 * csag * y ** 2
                return h * 1e9  # in nanometers
            elif self.mir_shape == self.ELLIPSE:
                if self.p == 0 or self.q == 0 or self.theta == 0:
                    raise Exception('Please enter P, Q, theta values')
                h = self.elp_heights(x, self.p, self.q, self.theta, 0, 0, 0)
                h = h * np.ones_like(y)  # Expand along sagittal
                return h * 1e9  # in nanometers
            elif self.mir_shape == self.OTHER:
                func = self.get_eqn(self.eqn_shape)
                if func is None:
                    raise Exception('Please enter an equation for shape')
                h = func(x, y)
                if self.noise_fe > 0.0:
                    h += np.random.normal(scale=np.sqrt(np.max(h)*self.noise_fe), size=h.shape)
                return h * 1e9  # in nanometers

    @staticmethod
    def elp_slopes(x, p, q, theta, center, rotate, piston):
        theta_si = theta * 1e-3
        center_si = center * 1e-3
        rotate_si = rotate * 1e-3
        x = x - center_si
        a = (p + q) / 2
        b = np.sqrt(p * q) * np.sin(theta_si)
        F = (1 / 2) * np.sqrt(p ** 2 + q ** 2 - 2 * p * q * np.cos(np.pi - (2 * theta_si)))
        alpha = np.arcsin(p / (2 * F) * np.sin(np.pi - (2 * theta_si)))
        mu = alpha - theta_si
        x0 = F - q * np.cos(alpha)
        # y0 = -b*np.sqrt(1-(x0/a)**2)

        sx_fit_fn = (np.cos(mu) ** 2) * (b / a) * (
                (np.cos(mu) * x + x0) /
                np.sqrt(-1 * (np.cos(mu) ** 2) * (x ** 2) - 2 * np.cos(mu) * x * x0 + (a ** 2) - (x0 ** 2))
        ) + np.cos(mu) * np.sin(-mu) + rotate_si
        sx_fit_fn = sx_fit_fn

        return sx_fit_fn

    @staticmethod
    def elp_heights(x, p, q, theta, center, rotate, piston):
        theta_si = theta * 1e-3
        center_si = center * 1e-3
        rotate_si = rotate * 1e-3
        piston_si = piston * 1e-9
        x = x - center_si
        a = (p + q) / 2
        b = np.sqrt(p * q) * np.sin(theta_si)
        F = (1 / 2) * np.sqrt(p ** 2 + q ** 2 - 2 * p * q * np.cos(np.pi - (2 * theta_si)))
        alpha = np.arcsin(p / (2 * F) * np.sin(np.pi - (2 * theta_si)))
        mu = alpha - theta_si
        x0 = F - q * np.cos(alpha)
        y0 = -b * np.sqrt(1 - (x0 / a) ** 2)

        z1 = np.cos(mu) * (-np.sqrt((b ** 2) * (1 - ((x * np.cos(mu) + x0) / a) ** 2)) - y0)
        z2 = np.sin(-mu) * x * np.cos(mu)
        z3 = np.cos(mu) * (y0 + np.sqrt((b ** 2) * (1 - (x0 / a) ** 2)))
        z_fit_fn = z1 + z2 + z3

        # rotate
        if rotate:
            # x        = x * np.cos(rotate_si) + z_fit_fn * np.sin(rotate_si)
            z_fit_fn = x * np.sin(rotate_si) + z_fit_fn * np.cos(rotate_si)
        if piston:
            z_fit_fn = z_fit_fn + piston_si

        z_fit_fn = z_fit_fn

        return z_fit_fn
