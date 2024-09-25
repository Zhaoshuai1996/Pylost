# coding=utf-8
"""OWH5File is copied from orange native OWFile and adapted to pylost use
"""
import logging
import os
from typing import List
from urllib.parse import urlparse
from pathlib import Path

import h5py
import silx
from Orange.widgets import gui, widget
from Orange.widgets.data.owfile import LineEditSelectOnFocus, NamedURLModel
from Orange.widgets.utils.signals import Output
from PyQt5 import QtWebEngineWidgets, QtCore, QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QGridLayout, QStackedWidget
from PyQt5.QtWidgets import \
    QStyle, QComboBox, QSizePolicy as Policy, QCompleter
from orangewidget.settings import Setting
from orangewidget.utils.filedialogs import RecentPathsWComboMixin, RecentPath, open_filename_dialog
from orangewidget.utils.signals import Input
from orangewidget.widget import Msg

from pylost_widgets.util.DataViewerFrameOrange import DataViewerFrameOrange
from pylost_widgets.util.ow_filereaders import H5Reader
from pylost_widgets.util.util_functions import get_entries, closeH5, get_instrument_details, get_stitch_setups, \
    get_data_from_h5, get_setup_from_h5

from pylost_widgets.widgets._PylostBase import PylostWidgets

log = logging.getLogger(__name__)


class OWFileH5(PylostWidgets, RecentPathsWComboMixin):
    """Widget to load data in standard hdf5 format created for PyLOSt"""
    name = 'Data (H5)'
    description = 'Loads data from h5 file'
    icon = "../icons/files.svg"
    priority = 12

    class Inputs:
        fname = Input('file_name', str, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    want_main_area = 1

    SEARCH_PATHS = []  # TODO: set default paths
    LOCAL_FILE, URL = range(2)

    recent_paths: List[RecentPath]
    recent_urls: List[str]

    recent_paths = Setting([])
    recent_urls = Setting([])
    source = Setting(LOCAL_FILE)
    sheet_names = Setting({})

    entryIdx = Setting(0)
    scanIdx = Setting(0)
    setupIdx = Setting(0)

    class Information(widget.OWWidget.Information):
        loaded_setup = Msg("Setup loaded successfully")

    class Warning(widget.OWWidget.Warning):
        file_too_big = Msg("The file is too large to load automatically. Press Reload to load.")
        load_warning = Msg("Read warning:\n{}")

    class Error(widget.OWWidget.Error):
        file_not_found = Msg("File not found.")
        url_not_found = Msg("URL not found.")
        missing_reader = Msg("Missing reader.")
        sheet_error = Msg("Error listing available sheets.")
        unknown = Msg("Read error:\n{}")

    class NoFileSelected:
        pass

    def __init__(self):
        super().__init__()
        self._clean_recentpaths()
        RecentPathsWComboMixin.__init__(self)
        self.h5Obj = None
        self.data_obj = {}
        self.add_path('X:/_mirrors/h5/temp.h5')

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout, addSpace=True)
        vbox = gui.radioButtons(None, self, "source", box=True, addSpace=True, callback=self.load_data,
                                addToLayout=False)

        rb_button = gui.appendRadioButton(vbox, "File:", addToLayout=False)
        layout.addWidget(rb_button, 0, 0, Qt.AlignVCenter)

        box = gui.hBox(None, addToLayout=False, margin=0)
        box.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.file_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.file_combo.activated[int].connect(self.select_file)
        box.layout().addWidget(self.file_combo)
        layout.addWidget(box, 0, 1)

        file_button = gui.button(None, self, '...', callback=self.browse_file, autoDefault=False)
        file_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(file_button, 0, 2)

        reload_button = gui.button(None, self, "Reload", callback=self.load_data, autoDefault=False)
        reload_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(reload_button, 0, 3)

        rb_button = gui.appendRadioButton(vbox, "URL:", addToLayout=False)
        layout.addWidget(rb_button, 1, 0, Qt.AlignVCenter)

        self.url_combo = url_combo = QComboBox()
        url_model = NamedURLModel(self.sheet_names)
        url_model.wrap(self.recent_urls)
        url_combo.setLineEdit(LineEditSelectOnFocus())
        url_combo.setModel(url_model)
        url_combo.setSizePolicy(Policy.Ignored, Policy.Fixed)
        url_combo.setEditable(True)
        url_combo.setInsertPolicy(url_combo.InsertAtTop)
        url_edit = url_combo.lineEdit()
        l, t, r, b = url_edit.getTextMargins()
        url_edit.setTextMargins(l + 5, t, r, b)
        layout.addWidget(url_combo, 1, 1, 3, 3)
        url_combo.activated.connect(self._url_set)
        # whit completer we set that combo box is case sensitive when
        # matching the history
        completer = QCompleter()
        completer.setCaseSensitivity(Qt.CaseSensitive)
        url_combo.setCompleter(completer)

        self.infoEntry = gui.widgetLabel(self.controlArea, "Measured by _____ instrument")
        self.selEntry = gui.comboBox(self.controlArea, self, "entryIdx", label='Select entry',
                                     callback=self.change_entry, orientation=Qt.Horizontal)
        self.selScan = gui.comboBox(self.controlArea, self, "scanIdx", label='Select scan', callback=self.change_scan,
                                    orientation=Qt.Horizontal, addSpace=True)
        self.selSetup = gui.comboBox(self.controlArea, self, "setupIdx", label='Load saved setup',
                                     callback=self.change_setup, orientation=Qt.Horizontal, addSpace=True)

        # H5 file viewer
        box = gui.vBox(self.controlArea, "File viewer")
        self.__treeview = silx.gui.hdf5.Hdf5TreeView(self)
        self.__treeview.activated.connect(self.displayData)
        box.layout().addWidget(self.__treeview)

        # Data viewers
        self.stack = QStackedWidget(self)
        box1 = gui.vBox(self.stack, "Data viewer")
        self.__dataViewer = DataViewerFrameOrange(self, show_mask=False)
        box1.layout().addWidget(self.__dataViewer)

        box2 = gui.vBox(self.stack, "Url viewer")
        QtWebEngineWidgets.QWebEngineProfile.defaultProfile().downloadRequested.connect(self.on_downloadRequested)
        self.__urlview = QWebEngineView()
        self.__urlview.loadFinished.connect(self.urlLoadFinish)
        box2.layout().addWidget(self.__urlview)

        self.stack.addWidget(box1)
        self.stack.addWidget(box2)
        self.mainArea.layout().addWidget(self.stack)

    def _clean_recentpaths(self):
        pathlist = []
        for i, item in enumerate(self.recent_paths):
            if i > 20:
                break
            if Path(item.abspath).exists():
                pathlist.append(item)
        self.recent_paths = pathlist

    def displayData(self):
        """Display data in treeview"""
        selected = list(self.__treeview.selectedH5Nodes())
        if len(selected) == 1:
            self.__dataViewer.setData(selected[0])

    def clearViewer(self):
        "Clear data viewer"
        self.__dataViewer.setData(None)
        self.__treeview.findHdf5TreeModel().clear()

    def close(self):
        """Close h5 object and close widget window"""
        closeH5(self.h5Obj)
        super().close()

    def sizeHint(self):
        return QSize(1000, 550)

    @Inputs.fname
    def set_fname(self, fname):
        """
        Linked to filename input channel.

        :param data: Input filename
        :type data: str
        """
        if fname is not None:
            if os.path.exists(fname):
                self.add_path(fname)
                self.source = self.LOCAL_FILE
                self.load_data()
                self.data_obj['file_name'] = fname
            else:
                self.data_obj['file_name'] = ''
                self.info.set_output_summary(self.info.NoInput)
                self.clearViewer()
                raise self.Error.file_not_found
        else:
            self.data_obj = {}
            self.clearViewer()
            self.clear_messages()

    def select_file(self, n):
        """Select a file from dropdown"""
        assert n < len(self.recent_paths)
        super().select_file(n)
        if self.recent_paths:
            self.source = self.LOCAL_FILE
            self.load_data()
            self.set_file_list()

    def _url_set(self):
        """Load file from a url"""
        url = self.url_combo.currentText()
        pos = self.recent_urls.index(url)
        url = url.strip()

        if not urlparse(url).scheme:
            url = 'http://' + url
            self.url_combo.setItemText(pos, url)
            self.recent_urls[pos] = url

        self.source = self.URL
        self.load_data()

    def browse_file(self):
        """Browse file"""
        start_file = self.last_path() or os.path.expanduser("~/")

        readers = [H5Reader]
        filename, reader, _ = open_filename_dialog(start_file, None, readers)
        # filename, filter = QFileDialog.getOpenFileName(self, "Select File", start_file,'*.h5')
        if not filename:
            return
        self.add_path(filename)
        if reader is not None:
            self.recent_paths[0].file_format = reader.qualified_name()

        self.source = self.LOCAL_FILE
        self.load_data()

    # Open a file, create data from it and send it over the data channel
    def load_data(self):
        """Load data from file"""
        # We need to catch any exception type since anything can happen in
        # file readers
        self.clear_messages()
        self.set_file_list()
        self.info.set_output_summary('Loading file...')
        QtWidgets.qApp.processEvents()

        if self.last_path() and not os.path.exists(self.last_path()):
            self.info.set_output_summary('File not found')
            return self.Error.file_not_found

        if self.source == self.LOCAL_FILE:
            self.stack.setCurrentIndex(0)
            closeH5(self.h5Obj)
            self.h5Obj = h5py.File(self.last_path(), 'r')
            self.clearViewer()
            self.__treeview.findHdf5TreeModel().insertFile(self.last_path())
            self.load_entries()
            self.Outputs.data.send(None)
            self.setStatusMessage('{}'.format(os.path.basename(self.last_path())))
            self.info.set_output_summary('File loaded')
        else:
            self.stack.setCurrentIndex(1)
            url = self.url_combo.currentText().strip()
            self.load_url(url)
            self.setStatusMessage('{}'.format(url))
            self.info.set_output_summary('Url loaded')
        return None

    def load_url(self, url):
        """Load file from url"""
        if url:
            try:
                qurl = QUrl(url)
                self.__urlview.load(qurl)
            except Exception as ex:
                log.exception(ex)
                self.no_output()
                return self.Error.url_not_found
        else:
            self.no_output()

    @QtCore.pyqtSlot("QWebEngineDownloadItem*")
    def on_downloadRequested(self, download):
        """Download from url"""
        try:
            url_str = str(download.url().toEncoded(), "utf-8")
            # urlreader = H5UrlReader(url_str)
            # suffix = reader.get_extension() # Problem with authentication redirection
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", '',
                                                            H5Reader.DESCRIPTION + '(*{})'.format(
                                                                ' *'.join(H5Reader.EXTENSIONS)))
            if path:
                suffix = QtCore.QFileInfo(path).suffix()
                if '.' + suffix in H5Reader.EXTENSIONS:
                    download.setPath(path)
                    download.accept()

                    self.add_path(path)
                    self.source = self.LOCAL_FILE
                    self.load_data()
                else:
                    return self.Error.unknown('Unknown file extension')
        except Exception as ex:
            log.exception(ex)
            self.no_output()
            return self.Error.file_not_found

    def urlLoadFinish(self, flag):
        pass

    def load_entries(self):
        """Clear and load entry fields from input h5 data"""
        self.selSetup.clear()
        self.selScan.clear()
        self.selEntry.clear()
        entries = get_entries(self.h5Obj)
        self.selEntry.addItems(entries)

    def change_entry(self):
        """Called after selecting an entry (NXentry class)"""
        self.clear_messages()
        self.info.set_output_summary('Loading...')
        QtWidgets.qApp.processEvents()
        try:
            self.progressBarInit()
        except Exception as e:
            self.Error.unknown(repr(e))
        try:
            if self.selEntry.currentIndex() != 0:
                curEntry = self.selEntry.currentText()
                details = get_instrument_details(self.h5Obj[curEntry])
                if details is not None:
                    self.instrId = details.instrId
                    self.infoEntry.setText("Measured by {} instrument".format(details.instrName))
                self.load_scans(curEntry)
                self.report_progress(10)
                self.change_scan()
                self.report_progress(50)
                self.load_setups(curEntry)

                try:
                    self.data_obj['entry'] = curEntry
                    self.data_obj['module'] = 'scan_data'
                    get_data_from_h5(self.h5Obj, curEntry, self.data_obj)
                    self.report_progress(90)
                    self.data_obj['comment_log'] = 'File {} loaded. \nEntry {} selected.'.format(self.last_path(),
                                                                                                 curEntry)
                    self.data_obj['filename'] = os.path.basename(self.last_path())
                    self.Outputs.data.send(self.data_obj)
                except:
                    self.data_obj['entry'] = ''
                    self.data_obj['scan_data'] = {}
                    self.Outputs.data.send(None)
                    return self.Error.unknown('Error loading scan data from h5 file')
            else:
                self.infoEntry.setText("Measured by _____ instrument")
                self.data_obj['entry'] = ''
                self.data_obj['scan_data'] = {}
                self.Outputs.data.send(None)
                self.__dataViewer.setData(None)
                self.selScan.clear()
                self.selSetup.clear()
            self.info.set_output_summary('Entry loaded')
        except Exception as e:
            self.Error.unknown(repr(e))
        try:
            self.progressBarFinished()
        except Exception as e:
            self.Error.unknown(repr(e))

    def report_progress(self, val):
        """Update progressbar"""
        try:
            self.progressBarSet(val)
        except Exception as e:
            self.Error.unknown(repr(e))

    def change_scan(self):
        """Called after selecting a scan in the UI, correspoinding to an entry"""
        curScan = self.selScan.currentText().strip()
        self.__dataViewer.setData(self.h5Obj[curScan])

    def load_scans(self, tag):
        """Load scans data"""
        self.selScan.clear()
        self.h5SignalsNX = []
        self.signalTxts = []
        self.h5CurData = self.h5Obj[tag] if tag != '' and (tag in self.h5Obj.keys()) else self.h5Obj
        # If the link is to a dataset, preview as it is and no need to change the current data tag
        if isinstance(self.h5CurData, h5py.Dataset):
            self.appendSignal(self.h5CurData.name, '')
        else:
            if self.checkSignal(self.h5CurData):  # If the link is to a group, check if this group has attribute @signal
                self.appendSignal(self.h5CurData.name, self.h5CurData.attrs['signal'])
            else:  # Check if the subgroups has attribute @signal
                self.h5CurData.visit(self.searchH5Attr)

        self.selScan.addItems(self.h5SignalsNX)

    def searchH5Attr(self, name):
        """Check if the selected data is NXdata class and has a signal attribute"""
        if 'NX_class' in self.h5CurData[name].attrs and self.h5CurData[name].attrs['NX_class'] == 'NXdata' and 'signal' in self.h5CurData[name].attrs:
            fullname = self.h5CurData[name].name
            self.appendSignal(fullname, self.h5CurData[name].attrs['signal'])

    def appendSignal(self, fullname, signal):
        """Store the signal corresponding to name"""
        self.h5SignalsNX.append(fullname)
        self.signalTxts.append(signal)

    @staticmethod
    def checkSignal(obj):
        """Check if the data is nxdata and signal existsin attributes"""
        return 'NX_class' in obj.attrs and obj.attrs['NX_class'] == 'NXdata' and 'signal' in obj.attrs

    def load_setups(self, entry):
        """TO be done. Load stitching setups"""
        setups = get_stitch_setups(self.h5Obj, entry, first='Select setup')
        if len(setups) == 1:
            setups[0] = 'No setups available'
        self.selSetup.clear()
        self.selSetup.addItems(setups)

    def change_setup(self):
        """To be done. Load data corresponding to a setup"""
        if self.selSetup.currentIndex() != 0:
            curSetup = self.selSetup.currentText()
            try:
                self.data_obj['load_setup'] = {'name': curSetup, 'data': get_setup_from_h5(self.h5Obj, curSetup)}
                self.Outputs.data.send(self.data_obj)
                return self.Information.loaded_setup
            except:
                self.data_obj['load_setup'] = {}
                self.Outputs.data.send(None)
                return self.Error.unknown('Error loading setup data from h5 file')
        else:
            self.data_obj['load_setup'] = {}
            self.Outputs.data.send(None)

    def no_output(self):
        """Clear if no input data"""
        self.clearViewer()
        self.Outputs.data.send(None)
        self.info.set_output_summary(self.info.NoOutput)
        return None
