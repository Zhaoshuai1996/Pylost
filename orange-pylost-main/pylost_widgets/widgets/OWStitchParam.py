# coding=utf-8
import concurrent.futures
import re
import sys
from functools import partial

import numpy as np
from Orange.widgets import gui
from Orange.widgets.utils.concurrent import FutureWatcher, ThreadExecutor, methodinvoke
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.widget import OWWidget
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, QThread, Qt, pyqtSlot
from PyQt5.QtWidgets import QDialog, QFormLayout, QGroupBox, QLabel, QMessageBox, QScrollArea, QSizePolicy as Policy
from orangewidget.settings import Setting
from orangewidget.widget import Msg
from silx.gui.plot import Plot1D

from PyLOSt.databases.gs_table_classes import Algorithms, ConfigParams, InputDispTypes, Instruments, \
    StitchSetupAlgoOptions, StitchSetupOptionsCommon, StitchSetupOptionsInstr
from pylost_widgets.config import config_params
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.Task import Task
from pylost_widgets.util.update_algo_path import UpdateAlgoPath
from pylost_widgets.util.util_functions import DEFAULT_DATA_NAMES, alertMsg, copy_items, get_stitch_step, load_axis_pix, \
    load_class, parseDefValue
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets


class OWStitchParam(PylostWidgets, PylostBase):
    name = 'Stitch Parameters'
    description = 'Stitch data with given parameters.'
    icon = "../icons/stitch.svg"
    priority = 31

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('stitched data', dict, default=True, auto_summary=False)
        data_ref = Output('extracted reference', dict, auto_summary=False)
        correctors = Output('correctors piston/pitch/roll', dict, auto_summary=False)
        data_corrected = Output('corrected subapertures', dict, auto_summary=False)

    want_main_area = 0
    module = Setting('', schema_only=True)
    step_x = Setting(0.0)
    step_y = Setting(0.0)

    class Error(OWWidget.Error):
        unknown = Msg("Error:\n{}")
        stitch = Msg("Failed to stitch: {}")
        get_options = Msg("Error loading options:\n{}")

    qsopts = StitchSetupOptionsCommon.select()
    for item in qsopts:
        val = parseDefValue(item.defVal)
        disp_item = InputDispTypes.selectBy(id=item.dispTypeID)[0]
        if type(val) == str:
            exec('soption_{}=Setting("{}")'.format(item.option, val.strip()))
        else:
            exec('soption_{}=Setting({})'.format(item.option, val))

    qaopts = StitchSetupAlgoOptions.select()
    tempArr = []
    for item in qaopts:
        if item.option in tempArr:
            continue
        tempArr.append(item.option)
        val = parseDefValue(item.defVal)
        disp_item = InputDispTypes.selectBy(id=item.dispTypeID)[0]
        if disp_item.dispType == 'L':
            vals = re.split(',', item.allVals)
            vals = ([x.strip() for x in vals])
            exec('aoption_{}_items=Setting({})'.format(item.option, vals))
        if type(val) == str:
            exec('aoption_{}=Setting("{}")'.format(item.option, val.strip()))
        else:
            exec('aoption_{}=Setting({})'.format(item.option, val))
    del tempArr

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True)
        self.btnPath = gui.button(box, self, 'Update algorithms', callback=self.updateAlgoPath, stretch=1,
                                  autoDefault=False)
        self.btnStitch = gui.button(box, self, 'Stitch', callback=self.stitchData, stretch=1)
        self.btnStitch.setEnabled(False)
        self.btnAbort = gui.button(box, self, 'Abort', callback=self.abort, stretch=1)
        self.btnAbort.hide()

        self.formBoxLayout = gui.hBox(self.controlArea, '', stretch=10)

        # Forms
        self.eArr = []
        self.eArrSettings = []
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.formBoxLayout.layout().addWidget(scroll)
        self.formSetupOptions = gui.widgetBox(None, "Stitching Options", orientation=QFormLayout, spacing=14,
                                              sizePolicy=Policy(Policy.Preferred, Policy.Maximum))
        scroll.setWidget(self.formSetupOptions)

        self.aArr = []
        self.aArrSettings = []
        self.groups = {}
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.formBoxLayout.layout().addWidget(scroll)
        self.formAlgoOptions = gui.widgetBox(None, "Algorithm Options", orientation=QFormLayout, spacing=14,
                                             sizePolicy=Policy(Policy.Preferred, Policy.Maximum))
        scroll.setWidget(self.formAlgoOptions)

        self.boxCost = gui.hBox(self.controlArea, 'Cost evaluation', stretch=10)
        self.cost_curve = Plot1D()
        self.cost_vals = []
        self.boxCost.layout().addWidget(self.cost_curve)
        self.boxCost.hide()

        self._task = None
        self._executor = ThreadExecutor()

        self.algos = None
        self.algoIds = [-1]
        self.curAlgoId = -1
        self.qalgos = None
        self.get_algorithms()

        self.algo_class_name = 'no algorithm set'
        self.algoObj = None

        self.eArrType = []
        self.qinsopts = None
        self.qalgoopts = None
        self.aArrType = []

    def abort(self):
        self.btnStitch.show()
        self.btnAbort.hide()
        self.info.set_output_summary('Aborted stitching')
        if self._task is not None:
            self.cancel()
        self.finProgressBar()

    def cancel(self):
        """
        Cancel the current task (if any).
        """
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            # disconnect the `_task_finished` slot
            self._task.watcher.done.disconnect(self._task_finished)
            self._task = None

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()

    def sizeHint(self):
        return QSize(1000, 550)

    @Inputs.data
    def set_data(self, data):
        if data is not None:
            for i, item in enumerate(data.values()):
                if isinstance(item, MetrologyData):
                    break
                self.clearForm()
            if len(item.dim_detector) < 3:
                QMessageBox.critical(self, 'Invalid data.', 'Input data should be a sequence of 2D subapertures.')
                self.clearForm()

        super().set_data(data, update_names=True, deepcopy=True)
        self.Outputs.data.send(None)
        self.Outputs.data_ref.send(None)
        self.Outputs.data_corrected.send(None)
        if data is None:
            self.clearForm()

    def load_data(self, multi=False):
        super().load_data()
        self.loadForm(self.data_in['instrument_id'] if 'instrument_id' in self.data_in else None)
        self.btnStitch.setEnabled(True)

    def update_comment(self, comment='', prefix=''):
        super().update_comment(comment, prefix='Stitch Parameters')

    def stitchData(self):
        self.clear_messages()
        self.setStatusMessage('')
        self.cost_vals = []
        self.boxCost.hide()
        size = 0
        for i, item in enumerate(self.DATA_NAMES):
            size = self.data_in[item].size
        if size > 20 * 1e6:
            ok = QMessageBox.question(self, 'Data size warning', 'Large data size.\nDo you want to stitch anyway?',
                                      QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes)
            if ok != QMessageBox.Yes:
                return
        self.info.set_output_summary('Stitching...')
        QtWidgets.qApp.processEvents()
        try:
            if self._task is not None:
                self.cancel()
            assert self._task is None
            copy_items(self.data_in, self.data_out, deepcopy=True, copydata=True)
            stitch_options = self.get_current_options()
            self.step_x = stitch_options['stitch_step_x'] if 'stitch_step_x' in stitch_options else self.step_x
            self.step_y = stitch_options['stitch_step_y'] if 'stitch_step_y' in stitch_options else self.step_y
            use_step_always = stitch_options[
                'use_stitch_step_ignore_motors'] if 'use_stitch_step_ignore_motors' in stitch_options else False
            load_axis_pix(self.data_out, self.step_x, self.step_y, use_step_always=use_step_always)

            qalgo = Algorithms.selectBy(algoName=stitch_options['stitch_algorithm'])[0]
            self.algo_class_name = qalgo.functionName
            qlocs = ConfigParams.selectBy(paramName='ALGO_LOC_STITCH')
            locs = []
            for it in qlocs:
                locs.append(it.paramValue)
            if any(locs):
                algoClass = load_class(self.algo_class_name, locs)
                if algoClass is None:
                    raise Exception('Class "{}" not found'.format(self.algo_class_name))
                self.algoObj = algoClass(stitch_options, data_in=self.data_out)

                self._task = task = Task()
                end_progressbar = methodinvoke(self, "finProgressBar", ())

                def callback():
                    if task.cancelled:
                        end_progressbar()
                        raise Exception('Aborted stitching')

                stitch_fun = partial(self.algoObj.stitch, callback=callback)

                self.algoObj.output.connect(self.set_stitch_data)
                self.algoObj.progress.connect(self.report_progress)
                self.algoObj.info.connect(self.update_info)
                self.algoObj.cost.connect(self.update_cost)
                self.btnStitch.hide()
                self.btnAbort.show()

                # data = self.algoObj.stitch()
                # self.set_stitch_data(data)

                self.startProgressBar()
                task.future = self._executor.submit(stitch_fun)
                task.watcher = FutureWatcher(task.future)
                task.watcher.done.connect(self._task_finished)

                if config_params.DEFAULT_CLOSE_WIDGETS_AFTER_APPLY:
                    self.close()
            else:
                self.Error.unknown('No stitching algorithms found')
        except Exception as e:
            self.data_out['stitch_data'] = {}
            self.setStatusMessage('')
            self.Outputs.data.send(None)
            self.btnStitch.show()
            self.btnAbort.hide()
            self.endProgressBar()
            return self.Error.stitch(repr(e))

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        """
        Parameters
        ----------
        f : Future
            The future instance holding the result of learner evaluation.
        """
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None
        self.endProgressBar()

        try:
            self.btnStitch.show()
            self.btnAbort.hide()
            # results = f.result()
        except Exception as e:
            return self.Error.stitch(repr(e))

    @pyqtSlot(float)
    def setProgressValue(self, value):
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    @pyqtSlot()
    def finProgressBar(self):
        assert self.thread() is QThread.currentThread()
        self.endProgressBar()

    def startProgressBar(self):
        try:
            self.progressBarInit()
        except Exception as e:
            self.Error.unknown(repr(e))

    def endProgressBar(self):
        try:
            self.progressBarFinished()
        except Exception as e:
            self.Error.unknown(repr(e))

    def set_stitch_data(self, data):
        self.set_data_by_module(self.data_out, self.module, data['stitched_scans'])
        # self.data_out['stitch_data']  = data
        # self.data_out['module'] = 'stitch_data'
        for key in data:
            if key != 'stitched_scans':
                self.data_out[key] = data[key]
        comment = f'Stitched data with {self.algo_class_name}'
        status = f'{self.algo_class_name}'
        if self.aoption_cor_reference_extract:
            status = status + ', reference extracted'
            comment = comment + ', reference extracted'
        else:
            self.data_out.pop('height_reference_extracted', None)  # FIX: sometimes a reference is present even when not asked (present within a pickle)
        self.update_comment(comment)
        self.setStatusMessage(status)
        self.info.set_output_summary('Finished stitching')
        data_ref = self.get_suplementary_output(options='reference_extracted', data=self.data_out, module=self.module)
        self.Outputs.data_ref.send(data_ref)
        correctors = self.get_suplementary_output(options=['piston', 'pitch', 'roll'], data=self.data_out,
                                                  module=self.module, keep_tag=True)
        self.Outputs.correctors.send(correctors)
        data_cor = self.get_suplementary_output(options='corrected', data=self.data_out, module=self.module)
        self.Outputs.data_corrected.send(data_cor)

        # Remove corrected subapertures, as they are too big and unnecessary in most cases. If needed access them through output 'Outputs.data_corrected'
        self.remove_suplementary_output(options='corrected', data=self.data_out, module=self.module)
        self.Outputs.data.send(self.data_out)

    @staticmethod
    def get_suplementary_output(options, data, module, keep_tag=False):
        ret_data = {}
        if not isinstance(options, (list, tuple, np.ndarray)):
            options = [options]

        if module == 'custom':
            for x in DEFAULT_DATA_NAMES:
                for option in options:
                    key = x + '_' + option
                    if key in data:
                        ret_data[key if keep_tag else x] = data[key]
        elif module == 'scan_data':
            scans = data['scan_data']
            ret_scans = {}
            for i, it in enumerate(scans):
                scan = scans[it]
                for x in DEFAULT_DATA_NAMES:
                    for option in options:
                        key = x + '_' + option
                        if key in scan:
                            if it not in ret_scans:
                                ret_scans[it] = {}
                            ret_scans[it][key if keep_tag else x] = scan[key]
            if len(ret_scans) > 0:
                ret_data['scan_data'] = ret_scans
        return ret_data if len(ret_data) > 0 else None

    @staticmethod
    def remove_suplementary_output(options, data, module):
        if not isinstance(options, (list, tuple, np.ndarray)):
            options = [options]
        if module == 'custom':
            for x in DEFAULT_DATA_NAMES:
                for option in options:
                    key = x + '_' + option
                    if key in data:
                        del data[key]
        elif module == 'scan_data':
            scans = data['scan_data']
            for i, it in enumerate(scans):
                scan = scans[it]
                for x in DEFAULT_DATA_NAMES:
                    for option in options:
                        key = x + '_' + option
                        if key in scan:
                            del scan[key]

    def report_progress(self, val):
        try:
            self.setProgressValue(val)
        except Exception as e:
            self.Error.unknown(repr(e))

    def update_cost(self, val):
        try:
            self.boxCost.show()
            self.cost_vals += [val]
            x = np.arange(len(self.cost_vals))
            self.cost_curve.addCurve(x, self.cost_vals, legend='cost_eval')
        except Exception:
            pass

    def update_info(self, val):
        try:
            self.info.set_output_summary(val)
            QtWidgets.qApp.processEvents()
        except Exception as e:
            self.Error.unknown(f'Error receiving updates ({e})')

    def get_current_options(self):
        if self.controls.soption_stitch_algorithm.currentIndex() == 0:
            raise self.Error.get_options('Please select a stitching algorithm')
        options = {}
        for i, item in enumerate(self.qinsopts):
            copt = item if type(item) is StitchSetupOptionsCommon else \
                StitchSetupOptionsCommon.selectBy(id=item.optionID)[0]
            options[copt.option] = getattr(self, 'soption_' + copt.option)
        aoptions = {}
        for i, it in enumerate(self.qalgoopts):
            aoptions[it.option] = getattr(self, 'aoption_' + it.option)
        options['algorithm_options'] = aoptions

        return options

    def clearForm(self):
        self.btnStitch.setEnabled(False)
        for i, it in enumerate(self.eArr):
            le = self.eArr[i]
            if self.eArrType[i] == 'E':
                le.setText('')
            elif self.eArrType[i] == 'C':
                le.setChecked(False)
            elif self.eArrType[i] == 'S':
                setattr(self, self.eArrSettings[i], le.itemText(0))
            else:
                le.setText('')
        # self.algos.setCurrentIndex(0)
        self.algoChange()

    def fillForm(self, params):
        for i, item in enumerate(self.qinsopts):
            copt = StitchSetupOptionsCommon.selectBy(id=item.optionID)[0]
            le = self.eArr[i]
            val = str(params[copt.option]).strip()
            if self.eArrType[i] == 'E':
                le.setText(val)
            elif self.eArrType[i] == 'C':
                le.setChecked(parseDefValue(val))
            elif self.eArrType[i] == 'S':
                setattr(self, 'soption_' + copt.option, val)
            else:
                le.setText(val)

        # update algorithm options
        algo_params = params['algorithm_options']
        for i, it in enumerate(self.qalgoopts):
            le = self.aArr[i]
            val = str(algo_params[it.option]).strip()
            if self.aArrType[i] == 'E':
                le.setText(val)
            elif self.aArrType[i] == 'C':
                le.setChecked(parseDefValue(val))
            elif self.aArrType[i] == 'S':
                setattr(self, 'aoption_' + it.option, val)
            else:
                le.setText(val)

    def clearOptions(self, stype):
        if stype == 'S':
            self.clearLayout(self.formSetupOptions.layout())
            for it in self.eArr:
                it.setParent(None)
        elif stype == 'A':
            if any(self.groups):
                for key in self.groups:
                    self.groups[key].setParent(None)
            self.clearLayout(self.formAlgoOptions.layout())
            for it in self.aArr:
                it.setParent(None)

    def checkstep(self):
        try:
            idx = self.eArrSettings.index('soption_use_stitch_step_ignore_motors')
            if get_stitch_step(self.data_in)[0] != np.around(self.soption_stitch_step_x, 3) or \
                    get_stitch_step(self.data_in)[1] != np.around(self.soption_stitch_step_y, 3):
                self.eArr[self.eArrSettings.index('soption_use_stitch_step_ignore_motors')].setChecked(True)
        except:
            pass

    def checkignoremotors(self):
        try:
            if not self.soption_use_stitch_step_ignore_motors:
                stepx = get_stitch_step(self.data_in)[0]
                if abs(stepx) < 0.0001:
                    stepx = self.step_x
                self.eArr[self.eArrSettings.index('soption_stitch_step_x')].setText('{:.3f}'.format(stepx))
                stepy = get_stitch_step(self.data_in)[1]
                if abs(stepy) < 0.0001:
                    stepy = self.step_y
                self.eArr[self.eArrSettings.index('soption_stitch_step_y')].setText('{:.3f}'.format(stepy))
            self.eArr[self.eArrSettings.index('soption_stitch_step_x')].setEnabled(self.soption_use_stitch_step_ignore_motors)
            self.eArr[self.eArrSettings.index('soption_stitch_step_y')].setEnabled(self.soption_use_stitch_step_ignore_motors)
        except:
            pass

    def loadForm(self, instrId):
        self.clearOptions('S')
        self.clearOptions('A')
        self.eArr = []
        self.eArrSettings = []
        self.eArrType = []

        if instrId is None:
            self.qinsopts = StitchSetupOptionsCommon.select()
        else:
            self.qinsopts = StitchSetupOptionsInstr.selectBy(instrID=Instruments.selectBy(instrId=instrId)[0].id)

        for item in self.qinsopts:
            if instrId is None:
                copt = item
            else:
                copt = StitchSetupOptionsCommon.selectBy(id=item.optionID)[0]
            lblTxt = copt.option
            if item.defValUnit != '':
                lblTxt += ' (' + item.defValUnit + ')'
            lblDesc = copt.optionDesc

            disp_item = InputDispTypes.selectBy(id=copt.dispTypeID)[0]
            lbl_width = 200
            if disp_item.dispType == 'E':  # Lineedit
                if copt.option == 'stitch_step_x':
                    stepx = get_stitch_step(self.data_in)[0]
                    if abs(stepx) < 0.0001:
                        stepx = self.step_x
                    lblTxt = 'Manual constant stitch step X (mm)'
                    le = gui.lineEdit(self.formSetupOptions, self, 'soption_' + copt.option, label=lblTxt,
                                      orientation=Qt.Horizontal, tooltip=lblDesc, labelWidth=lbl_width,
                                      callback=self.checkstep)
                    le.setText('{:.3f}'.format(stepx))
                elif copt.option == 'stitch_step_y':
                    stepy = get_stitch_step(self.data_in)[1]
                    if abs(stepy) < 0.0001:
                        stepy = self.step_y
                    lblTxt = 'Manual constant stitch step Y (mm)'
                    le = gui.lineEdit(self.formSetupOptions, self, 'soption_' + copt.option, label=lblTxt,
                                      orientation=Qt.Horizontal, tooltip=lblDesc, labelWidth=lbl_width,
                                      callback=self.checkstep)
                    le.setText('{:.3f}'.format(stepy))
                else:
                    le = gui.lineEdit(self.formSetupOptions, self, 'soption_' + copt.option, label=lblTxt,
                                      orientation=Qt.Horizontal, tooltip=lblDesc, labelWidth=lbl_width,
                                      callback=self.checkstep)
            elif disp_item.dispType == 'C':  # Checkbox
                if 'use_stitch_step_ignore_motors' in lblTxt:
                    if abs(get_stitch_step(self.data_in)[0]) > 0.0001 or abs(get_stitch_step(self.data_in)[1]) > 0.0001:
                        le = gui.checkBox(self.formSetupOptions, self, 'soption_' + copt.option, label='Force constant manual steps',
                                          labelWidth=lbl_width, callback=self.checkignoremotors)
                        self.checkignoremotors()
                else:
                    le = gui.checkBox(self.formSetupOptions, self, 'soption_' + copt.option, label=lblTxt,
                                      labelWidth=lbl_width)
            elif disp_item.dispType == 'S':  # Selectbox
                if copt.option == 'stitch_algorithm':
                    items = self.get_algorithms()
                    le = gui.comboBox(self.formSetupOptions, self, 'soption_' + copt.option, label=lblTxt,
                                      callback=self.algoChange, orientation=Qt.Horizontal, labelWidth=lbl_width,
                                      tooltip=lblDesc, sendSelectedValue=True, items=items)
                    self.algos = le
                    self.algoChange()
                else:
                    # items = tuple(re.split(',', copt.allVals)) #TODO: need to change the table StitchSetupOptionsCommon
                    le = gui.comboBox(self.formSetupOptions, self, 'soption_' + copt.option, label=lblTxt,
                                      orientation=Qt.Horizontal, tooltip=lblDesc, sendSelectedValue=True,
                                      labelWidth=lbl_width)
            else:
                le = gui.lineEdit(self.formSetupOptions, self, 'soption_' + copt.option, label=lblTxt,
                                  orientation=Qt.Horizontal, tooltip=lblDesc, labelWidth=lbl_width)
            self.eArr.append(le)
            self.eArrSettings.append('soption_' + copt.option)
            self.eArrType.append(disp_item.dispType)

    def get_algorithms(self):
        algorithms = ['Select algorithm']
        self.algoIds = [-1]
        self.curAlgoId = -1
        self.qalgos = Algorithms.select()
        for it in self.qalgos:
            algorithms.append(it.algoName)
            self.algoIds.append(it.id)
        return tuple(algorithms)

    def algoChange(self):
        try:
            if self.algos is not None:
                idx = self.algos.currentIndex()
                if idx == 0 :
                    self.algos.setCurrentText('global_optimize')
                    self.soption_stitch_algorithm = 'global_optimize'
                    idx = 2
                self.curAlgoId = self.algoIds[idx]
                self.updateAlgoLayout()
        except Exception as e:
            print(e)

    def updateAlgoLayout(self):
        self.clearOptions('A')
        # self.qalgoopts = StitchSetupAlgoOptions.selectBy(algoID=self.curAlgoId)
        self.qalgoopts = StitchSetupAlgoOptions.select(StitchSetupAlgoOptions.q.algoID == self.curAlgoId,
                                                       orderBy=[StitchSetupAlgoOptions.q.id,
                                                                StitchSetupAlgoOptions.q.groupItems])

        self.aArr = []
        self.aArrSettings = []
        self.aArrType = []
        self.groups = {}
        for item in self.qalgoopts:
            lblTxt = item.option
            # lblDesc = item.optionDesc
            # leditTxt = item.defVal

            if item.groupItems is not None:
                if item.groupItems not in self.groups:
                    self.groups[item.groupItems] = gui.vBox(self.formAlgoOptions, item.groupItems)
                parent = self.groups[item.groupItems]
            else:
                parent = self.formAlgoOptions

            disp_item = InputDispTypes.selectBy(id=item.dispTypeID)[0]
            lbl_width = 200
            if disp_item.dispType == 'E':  # Lineedit
                le = gui.lineEdit(parent, self, 'aoption_' + item.option, label=lblTxt, orientation=Qt.Horizontal,
                                  labelWidth=lbl_width)
            elif disp_item.dispType == 'C':  # Checkbox
                le = gui.checkBox(parent, self, 'aoption_' + item.option, label=lblTxt, labelWidth=lbl_width)
            elif disp_item.dispType == 'S':  # Selectbox
                items = re.split(',', item.allVals)
                items = tuple([x.strip() for x in items])
                le = gui.comboBox(parent, self, 'aoption_' + item.option, label=lblTxt, orientation=Qt.Horizontal,
                                  labelWidth=lbl_width, sendSelectedValue=True, items=items)
            elif disp_item.dispType == 'L':  # ListWidget
                le = gui.listBox(parent, self, 'aoption_' + item.option, labels='aoption_' + item.option + '_items',
                                 box=lblTxt, labelWidth=lbl_width)
            else:
                le = gui.lineEdit(parent, self, 'aoption_' + item.option, label=lblTxt, orientation=Qt.Horizontal,
                                  labelWidth=lbl_width)

            le.setObjectName(item.option)
            self.aArr.append(le)
            self.aArrType.append(disp_item.dispType)
            self.aArrSettings.append('aoption_' + item.option)

    @staticmethod
    def clearLayout(layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()
            layout.itemAt(i).widget().setParent(None)

    def openHelpStitch(self):
        d = QDialog()
        formSetupOptions = QGroupBox("Stitching Options", d)
        layout = QFormLayout()
        for item in self.qinsopts:
            copt = StitchSetupOptionsCommon.selectBy(id=item.optionID)[0]
            lblTxt = copt.option
            if item.defValUnit != '':
                lblTxt += ' (' + item.defValUnit + ')'
            lblTxt += ' : '
            lblDesc = copt.optionDesc
            lbl1 = QLabel(lblTxt)
            lbl2 = QLabel(lblDesc)
            layout.addRow(lbl1, lbl2)
            if copt.option == 'stitch_algorithm':
                qalgos = Algorithms.select()
                for it in qalgos:
                    lbl3 = QLabel(' - ' + it.algoName)
                    lbl4 = QLabel(it.algoDesc)
                    layout.addRow(lbl3, lbl4)

        formSetupOptions.setLayout(layout)
        d.setWindowTitle("Stitching options dialog")
        d.resize(800, 400)
        d.exec_()

    def openHelpAlgo(self):
        if self.curAlgoId == -1:
            alertMsg(title='Select algorithm', msg='Please select an algorithm in the stitching options')
            return
        d = QDialog()
        formAlgOptions = QGroupBox("Algorithm Options", d)
        layout = QFormLayout()
        for item in self.qalgoopts:
            lblTxt = item.option + ' : '
            lblDesc = item.optionDesc
            lbl1 = QLabel(lblTxt)
            lbl2 = QLabel(lblDesc)
            layout.addRow(lbl1, lbl2)

        formAlgOptions.setLayout(layout)
        d.setWindowTitle("Algorithm options dialog")
        d.resize(800, 800)
        d.exec_()

    def updateAlgoPath(self):
        try:
            dialog = UpdateAlgoPath()
            dialog.exec_()
            self.load_data()
        except Exception as e:
            print(e)
            raise self.Error.unknown(sys.exc_info()[1])
