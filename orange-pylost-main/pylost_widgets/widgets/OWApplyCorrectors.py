# coding=utf-8
# Load file / enter list of values for pitch/roll/piston/reference etc. before stitching.
# Show stitch correctors with stitching, to compare external values
import os

import numpy as np
from Orange.data import FileFormat, Table
from Orange.data.io import class_from_qualified_name
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QInputDialog, QSizePolicy as Policy, QTextEdit
from astropy.units import Quantity
from orangewidget.settings import Setting
from orangewidget.utils.filedialogs import open_filename_dialog
from orangewidget.widget import Msg
from scipy.interpolate import interp1d, interp2d

from PyLOSt.algorithms.util.util_fit import getXYGrid
from PyLOSt.algorithms.util.util_stitching import getCorrectionMap
from pylost_widgets.util.util_functions import arr_to_pix, copy_items, get_params_data, get_params_scan, \
    get_suplementary_output, remove_suplementary_output
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWApplyCorrectos(PylostWidgets, PylostBase):
    """Widget to apply external pitch, roll and piston correctors to subaperture data and output the stitched surface"""
    name = 'Apply Correctors'
    description = 'Load pitch / roll / piston / reference / calibration corrections and apply them, before stitching.'
    icon = "../icons/tilt.svg"
    priority = 53

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('stitched data', dict, default=True, auto_summary=False)
        correctors = Output('correctors piston/pitch/roll', dict, auto_summary=False)
        data_corrected = Output('corrected subapertures', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    unit_piston = Setting('nm', schema_only=True)
    unit_pitch = Setting('urad', schema_only=True)
    unit_roll = Setting('urad', schema_only=True)
    unit_x = Setting('mm', schema_only=True)
    unit_y = Setting('mm', schema_only=True)

    count = Setting(0, schema_only=True)
    step_x = Setting(0.0, schema_only=True)
    step_y = Setting(0.0, schema_only=True)
    start_x = Setting(0.0, schema_only=True)
    start_y = Setting(0.0, schema_only=True)
    interp_cor = Setting(False, schema_only=True)

    start_file = Setting('')

    class Error(widget.OWWidget.Error):
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)
        self.btnApply = gui.button(box, self, 'Apply correctors', callback=self.apply, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(self.controlArea, "Load piston / pitch / roll correctors")
        hbox = gui.hBox(box)
        gui.label(hbox, self, "Piston : ", labelWidth=50)
        self.te_piston = QTextEdit(None)
        hbox.layout().addWidget(self.te_piston)
        gui.lineEdit(hbox, self, 'unit_piston', '', placeholderText='unit', toolTip='Unit', controlWidth=50)
        gui.button(hbox, self, '...', callback=self.load_piston, autoDefault=False)
        hbox = gui.hBox(box)
        gui.label(hbox, self, "Pitch : ", labelWidth=50)
        self.te_pitch = QTextEdit(None)
        hbox.layout().addWidget(self.te_pitch)
        gui.lineEdit(hbox, self, 'unit_pitch', '', placeholderText='unit', toolTip='Unit', controlWidth=50)
        gui.button(hbox, self, '...', callback=self.load_pitch, autoDefault=False)
        hbox = gui.hBox(box)
        gui.label(hbox, self, "Roll : ", labelWidth=50)
        self.te_roll = QTextEdit(None)
        hbox.layout().addWidget(self.te_roll)
        gui.lineEdit(hbox, self, 'unit_roll', '', placeholderText='unit', toolTip='Unit', controlWidth=50)
        gui.button(hbox, self, '...', callback=self.load_roll, autoDefault=False)

        # Motor positions
        box = gui.vBox(self.controlArea, "Motor positions associated to correctors")
        gui.checkBox(box, self, 'interp_cor',
                     'Interpolate correctors to match motor positions associated with subaperture data?')
        gui.lineEdit(box, self, 'count', 'Number of positions : ', controlWidth=50, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=[self.change_step_x, self.change_step_y])
        hbox = gui.hBox(box)
        gui.label(hbox, self, "Motor X : ", labelWidth=50)
        vbox = gui.vBox(hbox)
        hhbox = gui.hBox(vbox, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(hhbox, self, 'step_x', 'Step X', controlWidth=50, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.change_step_x)
        gui.lineEdit(hhbox, self, 'start_x', 'Start X', controlWidth=50, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.change_start_x)
        gui.button(hhbox, self, 'Reload from data', callback=self.reload_mx, autoDefault=False)
        hhbox = gui.hBox(vbox)
        self.te_mx = QTextEdit(None)
        hhbox.layout().addWidget(self.te_mx)
        gui.lineEdit(hhbox, self, 'unit_x', '', placeholderText='unit', toolTip='Unit', controlWidth=50)
        gui.button(hhbox, self, '...', callback=self.load_mx, autoDefault=False)

        hbox = gui.hBox(box)
        gui.label(hbox, self, "Motor Y : ", labelWidth=50)
        vbox = gui.vBox(hbox)
        hhbox = gui.hBox(vbox, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.lineEdit(hhbox, self, 'step_y', 'Step Y', controlWidth=50, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.change_step_y)
        gui.lineEdit(hhbox, self, 'start_y', 'Start Y', controlWidth=50, orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.change_start_y)
        gui.button(hhbox, self, 'Reload from data', callback=self.reload_my, autoDefault=False)
        hhbox = gui.hBox(vbox)
        self.te_my = QTextEdit(None)
        hhbox.layout().addWidget(self.te_my)
        gui.lineEdit(hhbox, self, 'unit_y', '', placeholderText='unit', toolTip='Unit', controlWidth=50)
        gui.button(hhbox, self, '...', callback=self.load_my, autoDefault=False)

        self.mXArr = np.array([])
        self.mYArr = np.array([])
        self.corPiston = np.array([])
        self.corPitch = np.array([])
        self.corRoll = np.array([])

    def sizeHint(self):
        return QSize(800, 50)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_names=True)
        if data is None:
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        """Implementation in super class PylostBase is used. Load motors from input data, if available"""
        super().load_data()
        self.count, pix_sz, self.mXArr, self.mYArr = get_params_data(self.data_in)
        self.load_motors_xy()
        # self.apply()

    def load_motors_xy(self, mx=True, my=True):
        """
        Load motor positions

        :param mx: Load motor x, True/False
        :type mx: bool
        :param my: Load motor y, True/False
        :type my: bool
        """
        try:
            if isinstance(self.mXArr, Quantity):
                self.unit_x = str(self.mXArr.unit)
                self.unit_y = str(self.mYArr.unit)
            mXArr = self.mXArr.value if isinstance(self.mXArr, Quantity) else self.mXArr
            mYArr = self.mYArr.value if isinstance(self.mYArr, Quantity) else self.mYArr
            if mx:
                self.te_mx.setText(',\t'.join(['{:.4f}'.format(x) for x in mXArr]) if isinstance(mXArr, (
                    list, tuple, np.ndarray)) else '{:.4f}'.format(mXArr))
                self.step_x = np.round(np.mean(np.diff(mXArr)), 4) if isinstance(mXArr,
                                                                                 (list, tuple, np.ndarray)) and len(
                    mXArr) > 0 else 0.0
                self.start_x = np.round(mXArr[0], 4) if isinstance(mXArr, (list, tuple, np.ndarray)) and len(
                    mXArr) > 0 else 0.0
            if my:
                self.te_my.setText(',\t'.join(['{:.4f}'.format(x) for x in mYArr]) if isinstance(mYArr, (
                    list, tuple, np.ndarray)) else '{:.4f}'.format(mYArr))
                self.step_y = np.round(np.mean(np.diff(mYArr)), 4) if isinstance(mYArr,
                                                                                 (list, tuple, np.ndarray)) and len(
                    mYArr) > 0 else 0.0
                self.start_y = np.round(mYArr[0], 4) if isinstance(mYArr, (list, tuple, np.ndarray)) and len(
                    mYArr) > 0 else 0.0

        except Exception as e:
            self.Error.unknown(repr(e))

    def update_comment(self, comment, prefix=''):
        """Implementation in super class PylostBase is used"""
        super().update_comment(comment, prefix='Load correctors')

    def load_piston(self):
        """Load piston from a file"""
        mdata = self.load_file()
        if np.any(mdata):
            self.te_piston.setText(',\t'.join(['{:.4f}'.format(x) for x in mdata]))

    def load_pitch(self):
        """Load pitch from a file"""
        mdata = self.load_file()
        if np.any(mdata):
            self.te_pitch.setText(',\t'.join(['{:.4f}'.format(x) for x in mdata]))

    def load_roll(self):
        """Load roll from a file"""
        mdata = self.load_file()
        if np.any(mdata):
            self.te_roll.setText(',\t'.join(['{:.4f}'.format(x) for x in mdata]))

    def load_mx(self):
        """Load motor x from a file"""
        mdata = self.load_file()
        if np.any(mdata):
            self.te_mx.setText(',\t'.join(['{:.4f}'.format(x) for x in mdata]))

    def load_my(self):
        """Load motor y from a file"""
        mdata = self.load_file()
        if np.any(mdata):
            self.te_my.setText(',\t'.join(['{:.4f}'.format(x) for x in mdata]))

    def reload_mx(self):
        """Reload motor x positions"""
        self.load_motors_xy(my=False)

    def reload_my(self):
        """Reload motor y positions"""
        self.load_motors_xy(mx=False)

    def change_step_x(self):
        """Update motor x with uniform step"""
        if self.count > 0:
            mx = self.start_x + self.step_x * np.arange(self.count)
            self.te_mx.setText(',\t'.join(['{:.4f}'.format(x) for x in mx]))

    def change_step_y(self):
        """Update motor y with uniform step"""
        if self.count > 0:
            my = self.start_y + self.step_y * np.arange(self.count)
            self.te_my.setText(',\t'.join(['{:.4f}'.format(x) for x in my]))

    def change_start_x(self):
        """Update motor x start position"""
        mx = np.array([float(x) for x in self.te_mx.toPlainText().split(',')] if self.te_mx.toPlainText() != '' else [])
        if len(mx) > 0:
            mx = mx - mx[0] + self.start_x
            self.te_mx.setText(',\t'.join(['{:.4f}'.format(x) for x in mx]))

    def change_start_y(self):
        """Update motor y start position"""
        my = np.array([float(x) for x in self.te_my.toPlainText().split(',')] if self.te_my.toPlainText() != '' else [])
        if len(my) > 0:
            my = my - my[0] + self.start_y
            self.te_my.setText(',\t'.join(['{:.4f}'.format(x) for x in my]))

    def load_file(self):
        """
        Load piston/pitch/roll values from a csv or txt file.
        """
        mdata = []
        try:
            if self.start_file == '':
                self.start_file = os.path.expanduser("~/")
            filt = ['.csv', '.txt', '.pkl']
            readers = [f for f in FileFormat.formats
                       if getattr(f, 'read', None)
                       and getattr(f, "EXTENSIONS", None)
                       and any(set(getattr(f, "EXTENSIONS", None)).intersection(filt))]
            filename, reader_nm, _ = open_filename_dialog(self.start_file, None, readers)
            self.start_file = filename
            if os.path.exists(filename):
                if reader_nm is not None:
                    reader_class = class_from_qualified_name(reader_nm)
                    reader = reader_class(filename)
                else:
                    reader = FileFormat.get_reader(filename)
                data = reader.read()

                if isinstance(data, (Table, list, tuple, np.ndarray)):
                    mdata = np.array(data).ravel()
                elif isinstance(data, dict) and any(data):
                    item, ok = QInputDialog.getItem(self, 'Get correctors', 'Select motor item:',
                                                    [x for x in data if isinstance(data[x], np.ndarray)])
                    if ok:
                        mdata = data[item].ravel()
        except Exception as e:
            self.Error.unknown(repr(e))
        return mdata

    def update_motor_arrays(self):
        """Get motor and corrector values from field in the UI, and convert them to quantities"""
        self.mXArr = Quantity(
            [float(x) for x in self.te_mx.toPlainText().split(',')] if self.te_mx.toPlainText() != '' else [],
            self.unit_x)
        self.mYArr = Quantity(
            [float(x) for x in self.te_my.toPlainText().split(',')] if self.te_my.toPlainText() != '' else [],
            self.unit_y)
        self.corPiston = Quantity(
            [float(x) for x in self.te_piston.toPlainText().split(',')] if self.te_piston.toPlainText() != '' else [],
            self.unit_piston)
        self.corPitch = Quantity(
            [float(x) for x in self.te_pitch.toPlainText().split(',')] if self.te_pitch.toPlainText() != '' else [],
            self.unit_pitch)
        self.corRoll = Quantity(
            [float(x) for x in self.te_roll.toPlainText().split(',')] if self.te_roll.toPlainText() != '' else [],
            self.unit_roll)

    def apply(self):
        """Apply corrections and send corrected and stitched data"""
        try:
            self.update_motor_arrays()
            super().apply_scans()
            correctors = get_suplementary_output(options=['piston', 'pitch', 'roll'], data=self.data_out,
                                                 module=self.module, keep_tag=True)
            self.Outputs.correctors.send(correctors)
            data_cor = get_suplementary_output(options='corrected', data=self.data_out, module=self.module)
            self.Outputs.data_corrected.send(data_cor)

            # Remove corrected subapertures, as they are too big and unnecessary in most cases. If needed access them through output 'Outputs.data_corrected'
            remove_suplementary_output(options='corrected', data=self.data_out, module=self.module)
            self.Outputs.data.send(self.data_out)
        except Exception as e:
            self.Error.unknown(str(e))

    def apply_scan(self, scan, scan_name=None, comment=''):
        """Apply for each scan. Reimplemented from pylostbasae"""
        # If different scans have different motor positions in subaperture data, it is taken into account in interpolation for each scan
        nb_subaps, pix_sz, mXData, mYData = get_params_scan(scan)
        corPiston = self.corPiston
        corPitch = self.corPitch
        corRoll = self.corRoll
        if self.interp_cor:
            flag_x = len(mXData) > 0 and np.any(np.diff(mXData) > 0) and len(self.mXArr) != 0 and np.any(
                np.diff(self.mXArr) != 0)
            flag_y = len(mYData) > 0 and np.any(np.diff(mYData) > 0) and len(self.mYArr) != 0 and np.any(
                np.diff(self.mYArr) != 0)
            flag_2D = flag_x and flag_y
            if flag_2D:
                cx1 = interp2d(self.mXArr, self.mYArr, self.corPiston)
                cx2 = interp2d(self.mXArr, self.mYArr, self.corPitch)
                cx3 = interp2d(self.mXArr, self.mYArr, self.corRoll)
                corPiston = cx1(mXData, mYData)
                corPitch = cx2(mXData, mYData)
                corRoll = cx3(mXData, mYData)
            elif flag_x or flag_y:
                cx1 = interp1d(self.mXArr if flag_x else self.mYArr, self.corPiston)
                cx2 = interp1d(self.mXArr if flag_x else self.mYArr, self.corPitch)
                cx3 = interp1d(self.mXArr if flag_x else self.mYArr, self.corRoll)
                corPiston = cx1(mXData if flag_x else mYData)
                corPitch = cx2(mXData if flag_x else mYData)
                corRoll = cx3(mXData if flag_x else mYData)
        elif nb_subaps != self.count:
            raise Exception('Number of subapertures ({}) do not match tne number of positions ({}) entered here{}. '
                            'Please check interpolate or correct the position count'.format(nb_subaps, self.count,
                                                                                            ' for scan {}'.format(
                                                                                                scan_name) if scan_name is not None else ''))
        else:
            # If not interpolating and entering new motor positions of same length as nb_subaps, force to correct according to new values
            mXData = self.mXArr
            mYData = self.mYArr

        # Get motors in pixels
        if pix_sz is None or len(pix_sz) == 0:
            raise Exception('Pixel size is not available')
        mxp = arr_to_pix(mXData, pix_sz[-1], nb_subaps)[0]
        myp = arr_to_pix(mYData, pix_sz[-2 if len(pix_sz) > 1 else -1], nb_subaps)[0]
        correctors = np.zeros((len(mxp), 3), dtype=np.float32)
        correctors[:, 0] = corPiston
        correctors[:, 1] = corRoll
        correctors[:, 2] = corPitch

        scan_result = {}
        copy_items(scan, scan_result)
        pix_sz_val = [x.value if isinstance(x, Quantity) else x for x in pix_sz]
        for i, item in enumerate(self.DATA_NAMES):
            if item in scan:
                if item == 'slopes_x':
                    correctors[:, 0] = 0
                    correctors[:, 1] = 0
                if item == 'slopes_y':
                    correctors[:, 0] = 0
                    correctors[:, 2] = 0
                scan_result[item], scan_result[item + '_corrected'], comment = self.apply_item(scan[item], mxp, myp,
                                                                                               pix_sz_val, correctors,
                                                                                               comment=comment,
                                                                                               item=item)
                scan_result[item + '_piston'] = correctors[:, 0]
                scan_result[item + '_roll'] = correctors[:, 1]
                scan_result[item + '_pitch'] = correctors[:, 2]

        return scan_result, comment

    def apply_item(self, Z, mX, mY, pix_size, correctors, comment='', item=None):
        """
        Apply for each data item, e.g. height, slopes_x, slopes_y

        :param Z: height, slopes_x, slopes_y dataset
        :type Z: MetrologyData
        :param mX: Motor x
        :type mX: numpy.ndarray
        :param mY: Motor y
        :type mY: numpy.ndarray
        :param pix_size: Pixel size
        :type pix_size: list[astropy.Quantity]
        :param correctors: Correctors like piston, pitch, roll
        :type correctors: numpy.ndarray
        :param comment: Comment text
        :type comment: str
        :param item: Name of the item
        :type item: str
        :return: Stitched Z, Corrected Z, comment
        :rtype: (MetrologyData, MetrologyData, str)
        """
        dims = super().get_detector_dimensions(Z)
        axes = dims.nonzero()[0]
        if len(axes) != 2:
            raise Exception('Stitching currently implemented only for 2D images')

        Zcor = np.full(Z.shape, np.nan, dtype=Z.dtype)
        ny = Z.shape[axes[-2]]
        nx = Z.shape[axes[-1]]
        res_szX = nx + np.nanmax(mX)  # result size X stitch dirctn
        res_szY = ny + np.nanmax(mY)  # result size Y
        mask_mir = np.full((res_szY, res_szX), True, dtype=bool)
        xx, yy = getXYGrid(mask_mir, pix_size=pix_size, order=2, mask=mask_mir)
        res_item = np.full([res_szY, res_szX], np.nan)
        res_intensity = np.full([res_szY, res_szX], np.nan)

        for j, s in enumerate(Z):
            ox = mX[j]
            oy = mY[j]
            slc = (slice(oy, oy + ny), slice(ox, ox + nx))
            cor = getCorrectionMap(correctors[j, :], item, xx[slc], yy[slc])
            temp = (Z[j, :, :].value if isinstance(Z, Quantity) else Z[j, :, :]) + cor
            res_item[slc] = np.nansum(np.dstack((res_item[slc], temp)), 2)
            res_intensity[slc] = np.nansum([res_intensity[slc], (~np.isnan(temp)).astype(int)], axis=0)
            Zcor[j] = temp

        Zstitch = np.divide(res_item, res_intensity)
        Zstitch = Z.copy_to_detector_format(Zstitch)
        comment += 'Stitched {} subapertures. \n'.format(item)

        return Zstitch, Zcor, comment
