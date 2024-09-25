# coding=utf-8
import logging
import os
from functools import partial
from typing import List
from pathlib import Path

from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Output
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QDir, QDirIterator, QSize, QThread, Qt, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QGridLayout, QSizePolicy as Policy, QStyle
from orangewidget.settings import Setting
from orangewidget.utils.filedialogs import RecentPath, RecentPathsWComboMixin, format_filter
from orangewidget.widget import Msg
from silx.gui import qt
from silx.gui.data import DataViews

from pylost_widgets.util.DataViewerFrameOrange import DataViewerFrameOrange
from pylost_widgets.util.DictionaryTree import DictionaryTreeWidget
from pylost_widgets.util.FileSeqLoader import FileSeqLoader
from pylost_widgets.util.Task import Task
from pylost_widgets.util.resource_path import resource_path
from pylost_widgets.util.util_functions import copy_items, get_default_data_names, parseQInt

log = logging.getLogger(__name__)

qtCreatorFile = resource_path(os.path.join("gui", "dialog_import.ui"))  # Enter file here.
UI_import, QtBaseClass = uic.loadUiType(qtCreatorFile)

import concurrent.futures
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke
from pylost_widgets.util.ow_filereaders import *
from pylost_widgets.widgets._PylostBase import PylostWidgets


class OWFileScans(PylostWidgets, RecentPathsWComboMixin):
    """Widget to load scan data"""
    name = 'Data (scans)'
    description = 'Loads raw instrument format data and saves it to a standard format file (in hdf5 or pkl)'
    icon = "../icons/waves.svg"
    priority = 13

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    recent_paths: List[RecentPath]
    recent_paths = Setting([])

    want_main_area = True
    TIME, NAME, SORT_SELECTION = range(3)
    scan_count = Setting(1, schema_only=True)
    file_readers = Setting(-1, schema_only=True)
    enable_multiselect = Setting(False, schema_only=True)
    sort = Setting(TIME, schema_only=True)
    fname_prefix = Setting('', schema_only=True)
    fname_format = Setting('', schema_only=True)
    flip_order = Setting(False, schema_only=True)
    flip_scans = Setting('', schema_only=True)

    split_with_format_str = Setting(False, schema_only=True)
    include_subfolders = Setting(False, schema_only=True)

    class Warning(widget.OWWidget.Warning):
        file_too_big = Msg("The file is too large to load automatically. Press Reload to load.")
        load_warning = Msg("Read warning:\n{}")

    class Error(widget.OWWidget.Error):
        incorrect_input = Msg(
            "Invalid input data. Data type of input can only be in (dict, tuple, np.ndarray, Table(Orange)).")
        file_not_found = Msg("File not found.")
        missing_reader = Msg("Missing reader.")
        unknown = Msg("Read error:\n{}")

    class NoFileSelected:
        pass

    def __init__(self):
        super().__init__()
        self._clean_recentpaths()
        RecentPathsWComboMixin.__init__(self)
        self.data_out = {}
        self.data_in = {}
        self.source = 0
        self.filename_seq = []
        self.DEFAULT_DATA_NAMES = get_default_data_names()
        self.default_axis_names = ['Motor', 'Y', 'X']
        self.default_dim_detector = [-2, -1]
        if self.last_path() is not None and any(self.last_path()):
            self.add_path(self.last_path())

        self.total_file_count = 0
        self.all_file_names = []
        self.readers = []
        self.scan_files = []
        # self.controlArea.setMinimumWidth(400)

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, "Load", margin=10, orientation=layout, addSpace=True, stretch=1)
        lbl = gui.widgetLabel(None, 'Input folder: ')
        layout.addWidget(lbl, 0, 0)
        self.file_combo.setMinimumWidth(150)
        self.file_combo.setSizePolicy(Policy.Fixed, Policy.Fixed)
        self.file_combo.activated[int].connect(self.select_file)
        layout.addWidget(self.file_combo, 0, 1)

        self.file_button = gui.button(None, self, '...', callback=self.browse_folder, autoDefault=False)
        self.file_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(self.file_button, 0, 2)

        self.reload_button = gui.button(None, self, "Reload", callback=self.load_folder, autoDefault=False)
        self.reload_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(self.reload_button, 0, 3)

        lbl = gui.widgetLabel(None, 'Number of scans: ')
        layout.addWidget(lbl, 1, 0)
        le_scans = gui.lineEdit(None, self, 'scan_count', 'Number of scans')
        le_scans.setMinimumWidth(50)
        le_scans.setSizePolicy(Policy.Fixed, Policy.Fixed)
        le_scans.returnPressed.connect(self.preview_names)
        layout.addWidget(le_scans, 1, 1)

        lbl = gui.widgetLabel(None, 'Select file reader: ')
        layout.addWidget(lbl, 2, 0)
        cb_readers = gui.comboBox(None, self, 'file_readers', label='File reader',
                                  callback=[self.load_default_reader_fname_format, self.preview_names])
        cb_readers.setMinimumWidth(300)
        cb_readers.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(cb_readers, 2, 1, 1, 3)

        self.lbl_file_count = gui.widgetLabel(self.controlArea, '')

        rb = gui.radioButtonsInBox(self.controlArea, self, "sort", label='Sort',
                                   btnLabels=['Time', 'Name', 'Sort selection'],
                                   box=True, addSpace=True, callback=self.preview_names)

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, "Sort selection", margin=10, orientation=layout, addSpace=True, stretch=1)
        lbl = gui.widgetLabel(None, 'First file: ')
        layout.addWidget(lbl, 0, 0)
        self.lbl_fname = gui.widgetLabel(None, '')
        self.lbl_fname.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(self.lbl_fname, 0, 1)
        lbl = gui.widgetLabel(None, 'Format string (e.g. |_subap_|.ext) and Prefix: ')
        layout.addWidget(lbl, 1, 0, 1, 2)
        hbox = gui.hBox(None)
        fmt_prefix = gui.lineEdit(hbox, self, 'fname_prefix', 'Prefix : ', sizePolicy=(Policy.Fixed, Policy.Fixed))
        fmt_fname = gui.lineEdit(hbox, self, 'fname_format', 'Format string : ',
                                 sizePolicy=(Policy.Fixed, Policy.Fixed))
        fmt_prefix.editingFinished.connect(self.preview_names)
        fmt_fname.editingFinished.connect(self.preview_names)
        fmt_prefix.editingFinished.connect(self.update_sort)
        fmt_fname.editingFinished.connect(self.update_sort)
        layout.addWidget(hbox, 2, 0, 1, 2)

        checkbox = gui.checkBox(None, self, "split_with_format_str",
                                "Split scans using format string (last '|' for subaps)",
                                sizePolicy=(Policy.Fixed, Policy.Fixed),
                                callback=[self.preview_names, self.update_sort])
        layout.addWidget(checkbox, 3, 0, 1, 2)

        box = gui.vBox(self.controlArea, 'Subfloder scans')
        gui.checkBox(box, self, "include_subfolders", "Include subfolders", sizePolicy=(Policy.Fixed, Policy.Fixed),
                     callback=self.preview_names)

        box = gui.vBox(self.controlArea, 'Flip')
        gui.checkBox(box, self, "flip_order", "Flip subaperture sequential order in scans",
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.preview_names)
        gui.lineEdit(box, self, 'flip_scans', 'Scans to flip (e.g.0-3,8,9-11) :', orientation=Qt.Horizontal,
                     sizePolicy=(Policy.Fixed, Policy.Fixed), callback=self.preview_names)

        box = gui.hBox(self.controlArea)
        self.btnAbort = gui.button(box, self, 'Abort', callback=self.abort, stretch=1, autoDefault=False,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.btnLoad = gui.button(box, self, 'Load scans', callback=self.load_raw_data, autoDefault=False,
                                  sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.btnAbort.hide()

        # Tree and Data viewer
        box = gui.vBox(self.controlArea, "Data tree", stretch=18)
        self.__treeViewer = DictionaryTreeWidget(self, None)
        self.__treeViewer.itemClicked.connect(self.displayDataOut)
        box.layout().addWidget(self.__treeViewer)
        self.__treeViewer.parent().hide()

        box = gui.vBox(self.mainArea, "Preview", stretch=1)
        self.dataViewer = DataViewerFrameOrange(self, editInfo=True, show_mask=False)
        box.layout().addWidget(self.dataViewer)
        self.Outputs.data.send(None)

        self._task = None
        self._executor = ThreadExecutor()

        self.table = None

    def sizeHint(self):
        return QSize(1000, 600)

    def _clean_recentpaths(self):
        pathlist = []
        for i, item in enumerate(self.recent_paths):
            if i > 20:
                break
            if Path(item.abspath).exists():
                pathlist.append(item)
        self.recent_paths = pathlist

    def select_file(self, n):
        """Load from folder selected through dropdown"""
        assert n < len(self.recent_paths)
        super().select_file(n)
        if self.recent_paths:
            self.load_folder()
            self.set_file_list()

    def load_default_reader_fname_format(self):
        """Some prefix and filename formats for known file readers. Need to shift this functionality to inside the reader"""
        if self.readers[self.file_readers].__name__ == 'SharperReader':
            self.fname_prefix = 'data_'
            self.fname_format = '|_|_index_|.has'
        elif self.readers[self.file_readers].__name__ == 'MetroProReader':
            self.fname_format = '|-P|.dat'
        elif self.readers[self.file_readers].__name__ == 'VeecoReader':
            self.fname_format = '|_|.opd'
        else:
            self.fname_format = ''

    def browse_folder(self):
        """Browse folder"""
        # dg = QFileDialog()
        # dg.setFileMode(QFileDialog.Directory)
        # dg.setOption(QFileDialog.DontUseNativeDialog, True)
        # dg.setOption(QFileDialog.ShowDirsOnly, False)
        # dg.exec()
        # input_folder = dg.directory().absolutePath()
        start_loc = self.last_path() or os.path.expanduser("~/")
        input_folder = QFileDialog.getExistingDirectory(self, 'Load scan directory', start_loc)
        self.add_path(input_folder)
        self.load_folder()

    def load_folder(self):
        """Load filenames in the folder. Only known filereaders are used. Reader is added to a dropdown, and filenames for that reader are shown in dataviewer"""
        if self.last_path() != '' and os.path.exists(self.last_path()):
            self.info.set_output_summary('Loading...')
            self.__treeViewer.parent().hide()
            self.clearOutputData()
            dir = QDir(self.last_path())
            self.total_file_count = dir.count()
            self.all_file_names = dir.entryList()
            extensions = [os.path.splitext(x)[1] for x in self.all_file_names]
            extensions = list(dict.fromkeys(extensions))
            self.readers = [f for f in FileFormat.formats
                            if getattr(f, 'read', None)
                            and getattr(f, "EXTENSIONS", None)
                            and any(set(f.EXTENSIONS).intersection(extensions))]
            filters = [format_filter(f) for f in self.readers]
            self.controls.file_readers.clear()
            self.controls.file_readers.addItems(filters)
            self.load_default_reader_fname_format()
            self.preview_names()
            self.info.set_output_summary('Loaded file names from: {}'.format((self.last_path())))

    def update_sort(self):
        """Update sort, set to 2"""
        self.sort = 2

    def preview_names(self):
        """PReview names in the data viewer"""
        try:
            self.clear_messages()
            idx = self.controls.file_readers.currentIndex()
            if idx != -1:
                dir = QDir(self.last_path())
                data = self.preview_dir(dir, idx)
                if data is not None:
                    self.scan_files = np.asarray(
                        [[os.path.join(self.last_path(), y.decode('utf-8')) for y in x] for x in data], dtype='U')
                if self.include_subfolders:
                    it = QDirIterator(self.last_path(), QDirIterator.Subdirectories)
                    while it.hasNext():
                        dir_path = it.next()
                        if dir_path.endswith('.'):
                            continue
                        if not os.path.isdir(dir_path):
                            continue
                        dir = QDir(dir_path)
                        try:
                            new_data = self.preview_dir(dir, idx)
                        except Exception as e:
                            print(e)
                            new_data = np.array([])
                        if len(new_data) == 0:
                            continue

                        if data is None:
                            data = new_data
                            self.scan_files = np.asarray(
                                [[os.path.join(dir_path, y.decode('utf-8')) for y in x] for x in data], dtype='U')
                        else:
                            if new_data.shape[-1] > data.shape[-1]:
                                data = np.pad(data, [(0, 0), (0, new_data.shape[-1] - data.shape[-1])], mode='constant',
                                              constant_values='')
                                self.scan_files = np.pad(self.scan_files,
                                                         [(0, 0), (0, new_data.shape[-1] - self.scan_files.shape[-1])],
                                                         mode='constant', constant_values='')
                            elif new_data.shape[-1] < data.shape[-1]:
                                new_data = np.pad(new_data, [(0, 0), (0, data.shape[-1] - new_data.shape[-1])],
                                                  mode='constant', constant_values='')
                            data = np.vstack((data, new_data))
                            new_scans = np.asarray(
                                [[os.path.join(dir_path, y.decode('utf-8')) for y in x] for x in data], dtype='U')
                            self.scan_files = np.vstack((self.scan_files, new_scans))
                self.dataViewer.setData(data)
                self.update_table_menu()
                self.lbl_file_count.setText(
                    'Number of filenames in preview shape: {}, non-empty fields: {}'.format(data.shape,
                                                                                            data[data != ''].size))
        except Exception as e:
            print(e)
            self.Error.unknown(repr(e))

    def update_table_menu(self):
        """Add additional options such as delete item in the tableview"""
        try:
            view = self.dataViewer.getViewFromModeId(modeId=DataViews.RAW_MODE)
            if view is not None and self.table is None:
                from silx.gui.widgets.TableWidget import TableView
                widget = view.getWidget()
                self.table = widget.findChild(TableView)
                self.table.copySelectedCellsAction.triggered.connect(self.update_prefix)
                self.delAction = DeleteAction(self.table)
                # self.delAction.setText('Delete item')
                self.delAction.triggered.connect(self.del_file_from_list)
                self.table.addAction(self.delAction)
                QtWidgets.qApp.processEvents()
        except Exception as e:
            print(e)

    def del_file_from_list(self):
        """Delete action for a table ite"""
        selected_idx = self.table.selectedIndexes()
        if not selected_idx:
            return
        selected_idx_tuples = [(idx.row(), idx.column()) for idx in selected_idx]

        selected_rows = [idx[0] for idx in selected_idx_tuples]
        selected_columns = [idx[1] for idx in selected_idx_tuples]

        data_model = self.table.model()

        for row in range(min(selected_rows), max(selected_rows) + 1):
            for col in range(min(selected_columns), max(selected_columns) + 1):
                index = data_model.index(row, col)
                data_model.setData(index, "", role=qt.Qt.EditRole)
                self.scan_files[row, col] = ''

        self.lbl_file_count.setText(
            'Number of filenames in preview shape: {}, non-empty fields: {}'.format(self.scan_files.shape,
                                                                                    self.scan_files[
                                                                                        self.scan_files != ''].size))

    def update_prefix(self):
        """Update file name prefix, and filter names based on that"""
        qapp = qt.QApplication.instance()
        prefix = qapp.clipboard().text()
        if self.fname_format != '':
            fmt = self.fname_format
            if fmt.startswith('|'):
                fmt = fmt[1:]
            prefix = re.split(fmt, prefix)[0]
            if prefix == '':
                prefix = qapp.clipboard().text()
        self.fname_prefix = prefix

    @staticmethod
    def validate_prefix(prefix, x):
        """Validation for prefix, i.e. if it starts with %, finame with prefix string anywhere, other wise only at the start of the filename"""
        if prefix.startswith('%'):
            return prefix[1:] in x.decode('utf-8')
        else:
            return x.decode('utf-8').startswith(prefix)

    def preview_dir(self, dir, idx):
        """PReview directory"""
        data = None
        arr_fmt = np.array([])
        flist = dir.entryList(['*{}'.format(x) for x in self.readers[idx].EXTENSIONS])
        finfolist = dir.entryInfoList(['*{}'.format(x) for x in self.readers[idx].EXTENSIONS])
        flist = [x.encode('utf-8') for x in flist]
        file_names = np.array(flist)
        # Update filenames if any prefix is entered
        if self.fname_prefix != '':
            prefix = self.fname_prefix
            idx_arr = np.array([i for i, x in enumerate(file_names) if self.validate_prefix(prefix, x)])
            if len(idx_arr) == 0:
                return []
            file_names = file_names[idx_arr]
            finfolist = np.array(finfolist)[idx_arr]
        else:
            prefix = '%'
        if len(file_names) == 0:
            raise Exception('No filenames found. Please update prefix.')
        self.lbl_fname.setText(str(file_names[0].decode('utf-8')))

        if self.sort == self.TIME:
            sidx = np.argsort([x.lastModified() for x in finfolist])
            file_names = file_names[sidx]
        elif self.sort == self.NAME:
            file_names = np.sort(file_names)
        elif self.sort == self.SORT_SELECTION and self.fname_format != '':
            try:
                arr_fmt = [
                    [self.str_to_float(y) for y in re.split(prefix + self.fname_format, x.decode('utf-8')) if y != '']
                    for x in file_names]
                arr_fmt = np.asarray(arr_fmt, dtype=object)
                arr_lex = tuple([arr_fmt[:, i] for i in np.arange(arr_fmt.shape[1])[::-1]])
                file_names = file_names[np.lexsort(arr_lex)]
            except Exception as e:
                self.Error.unknown('Unable to parse. Maybe change prefix or format string. Error:{}'.format(e))

        # Split scans
        if self.split_with_format_str and np.any(arr_fmt) and arr_fmt.shape[-1] > 1:
            tmp_arr = []
            tmp_cnt = []
            for j in range(arr_fmt.shape[-1] - 1):
                tmp = np.unique(arr_fmt[:, j])
                tmp_arr += [tmp]
                tmp_cnt += [len(tmp)]

            for idx in np.ndindex(tuple(tmp_cnt)):
                scan_fmt = (prefix + self.fname_format).rsplit('|', 1)[0]
                for i in range(len(idx)):
                    val = tmp_arr[i][idx[i]]
                    val = val if isinstance(val, str) else '{:.0f}'.format(val)
                    scan_fmt = scan_fmt.replace('|', val, 1)
                new_scan = np.array([x for x in file_names if x.decode('utf-8').startswith(scan_fmt)])
                if len(new_scan) == 0:
                    continue
                if data is None:
                    data = new_scan
                else:
                    if new_scan.shape[-1] > data.shape[-1]:
                        data = np.pad(data, [(0, 0), (0, new_scan.shape[-1] - data.shape[-1])], mode='constant',
                                      constant_values='')
                    elif new_scan.shape[-1] < data.shape[-1]:
                        new_scan = np.pad(new_scan, [(0, data.shape[-1] - new_scan.shape[-1])], mode='constant',
                                          constant_values='')
                    data = np.vstack((data, new_scan))
            if data is None:
                data = np.asarray(file_names).reshape(self.scan_count, -1)
        elif self.scan_count > 1 and self.scan_count <= len(file_names):
            blocks = int(len(file_names) / self.scan_count)
            data = file_names[:self.scan_count * blocks].reshape(self.scan_count, -1)
        else:
            self.scan_count = 1
            data = np.asarray(file_names).reshape(self.scan_count, -1)

        if self.flip_scans == '':
            self.flip_scans = '0-{}'.format(self.scan_count)
        if self.flip_order:
            q = parseQInt(self.flip_scans)
            data[q] = data[q, ::-1]
            # data_flip = eval('data[' + self.flip_scans + ']')
            # exec('data[' + self.flip_scans + '] = data_flip[:, ::-1]')
        return data

    @staticmethod
    def str_to_float(y):
        """Convert string to float"""
        try:
            return float(y)
        except Exception:
            return y

    def load_raw_data(self):
        """Load data from selected files, as scans. Each row in the viewer is a scan with number of subaperture as column count"""
        try:
            self.clear_messages()
            if len(self.scan_files) > 0 and self.scan_files.ndim == 2:
                if self._task is not None:
                    self.cancel()
                assert self._task is None

                self.data_in = {}
                self.setStatusMessage('')
                self.clear_messages()
                self.reader = FileFormat.get_reader(self.scan_files[0, 0])

                seq_cls = FileSeqLoader()
                seq_cls.progress.connect(self.report_progress)
                self._task = task = Task()
                end_progressbar = methodinvoke(self, "finProgressBar", ())

                def callback():
                    if task.cancelled:
                        end_progressbar()
                        raise Exception('Aborted')

                load_fun = partial(seq_cls.load_scans, callback=callback, scan_files=self.scan_files,
                                   reader=self.reader)

                self.startProgressBar()
                task.future = self._executor.submit(load_fun)
                task.watcher = FutureWatcher(task.future)
                task.watcher.done.connect(self._task_finished)
                self.btnLoad.setEnabled(False)
                self.btnAbort.show()
        except Exception as e:
            self.endProgressBar()
            self.Error.unknown(repr(e))

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        """" Callback after loading scans is fininshed"""
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None
        self.endProgressBar()
        try:
            self.btnLoad.setEnabled(True)
            self.btnAbort.hide()

            self.data_in = f.result()
            if any(self.data_in):
                self.clearOutputData()
                self.format_output_data()
                self.__treeViewer.parent().show()
                self.__treeViewer.updateDictionary(self.data_out)
                self.Outputs.data.send(self.data_out)
                self.setStatusMessage('{}'.format(self.last_path()))
        except Exception as e:
            return self.Error.unknown(repr(e))

    def format_output_data(self):
        """Format output data  into standardized format, if it is availbale in the file reader"""
        len_scans = -1
        if hasattr(self.reader, 'data_standard_format'):
            if any(self.data_in):
                ret_data = {}
                scans = {}
                for key in self.data_in:
                    scan = self.reader.data_standard_format(self.data_in[key])
                    scans[key] = scan
                if hasattr(self.reader, 'PARAMS'):
                    if 'pixel_size' in self.reader.PARAMS:
                        ret_data['pix_size'] = self.reader.PARAMS['pixel_size']
                    if 'instr_scale_factor' in self.reader.PARAMS:
                        ret_data['instr_scale_factor'] = self.reader.PARAMS['instr_scale_factor']
                ret_data['scan_data'] = scans
                len_scans = len(scans)
                self.data_out = ret_data
        else:
            copy_items(self.data_in, self.data_out)
        self.data_out['module'] = 'scan_data'
        self.data_out['comment_log'] = 'Loaded {} scans from {}'.format(len_scans if len_scans > 0 else '',
                                                                        self.last_path())

    def abort(self):
        """Abort scans loading action"""
        self.clearViewer()
        self.btnLoad.setEnabled(True)
        self.btnAbort.hide()
        self.info.set_output_summary('Aborted loading')
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

    def displayDataOut(self):
        """Display output data"""
        data = self.__treeViewer.get_selected_data()
        self.dataViewer.setData(data)

    def clearViewer(self):
        """Clear data viewer"""
        self.__treeViewer.clearData()
        self.dataViewer.setData(None)

    def close(self):
        # TODO: close any opened files
        super().close()

    def clearOutput(self):
        """Cear output and send none"""
        self.clearOutputData()
        self.Outputs.data.send(None)

    def clearOutputData(self):
        """" cLear output data"""
        self.__treeViewer.clearData()
        self.dataViewer.setData(None)
        self.data_out = {}

    def report_progress(self, val):
        """Update progressbar"""
        try:
            self.setProgressValue(val)
        except Exception as e:
            self.Error.unknown(repr(e))

    @pyqtSlot(float)
    def setProgressValue(self, value):
        """Update progressbar slot"""
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    @pyqtSlot()
    def finProgressBar(self):
        """End progressbar"""
        assert self.thread() is QThread.currentThread()
        self.endProgressBar()

    def startProgressBar(self):
        """Start progressbar"""
        try:
            self.progressBarInit()
        except Exception as e:
            self.Error.unknown(repr(e))

    def endProgressBar(self):
        """End progressbar"""
        try:
            self.progressBarFinished()
        except Exception as e:
            self.Error.unknown(repr(e))


class DeleteAction(qt.QAction):
    """
    Called with delete action of an item in the table

    :param table: :class:`QTableView` to which this action belongs.
    """

    def __init__(self, table):
        if not isinstance(table, qt.QTableView):
            raise ValueError('DeleteAction must be initialised ' +
                             'with a QTableWidget.')
        super(DeleteAction, self).__init__(table)
        self.setText("Delete Item")
