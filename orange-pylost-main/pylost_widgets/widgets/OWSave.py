import os
from typing import List

import numpy as np
from Orange.data.io import FileFormat, class_from_qualified_name
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input
from Orange.widgets.widget import OWWidget
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QGridLayout, QSizePolicy as Policy, QAbstractItemView, QSplitter, QStyle, QInputDialog
from orangewidget.settings import Setting
from orangewidget.utils.filedialogs import RecentPathsWComboMixin, RecentPath, open_filename_dialog_save
from orangewidget.widget import Msg

from pylost_widgets.util.DictionaryTree import DictionaryTreeWidget
from pylost_widgets.util.util_functions import copy_items, questionMsgAdv, questionMsg
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets

from pathlib import Path


class OWSaveData(PylostWidgets, RecentPathsWComboMixin, PylostBase):
    name = 'Save Data'
    description = 'Save data to a file.'
    icon = "../icons/save.svg"
    priority = 14

    class Inputs:
        data = Input('data', dict, multiple=True, auto_summary=False)

    want_main_area = 0
    save_updates = Setting(False, schema_only=True)

    recent_paths: List[RecentPath]
    recent_paths = Setting([])
    enable_multiselect = Setting(False, schema_only=True)
    id_name_map = Setting({}, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        file_not_found = Msg("File not found.")
        missing_reader = Msg("Missing reader.")
        file_creation = Msg("Error creating file.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        self._clean_recentpaths()
        RecentPathsWComboMixin.__init__(self)
        PylostBase.__init__(self)
        self.loaded_savefile = False
        self.reader = None
        self.selitem = None
        self.selected_group = ''
        self.data_in_sel = {}
        self.load_flag = False
        self.add_path(os.path.expanduser("~/"))

        topbox = gui.vBox(None)
        ibox = gui.hBox(topbox, "Info", stretch=1)
        self.info_scroll(ibox)

        box = gui.vBox(topbox, 'Options')
        gui.checkBox(box, self, 'save_updates', 'Automatically save updates from workflow')
        self.btnSaveSimple = gui.button(None, self, 'Save all', callback=self.save_all, autoDefault=False, stretch=1,
                                        sizePolicy=(Policy.Fixed, Policy.Fixed))
        box.layout().addWidget(self.btnSaveSimple, alignment=Qt.AlignRight)

        # cbox = gui.vBox(None, 'Save advanced', margin=4, stretch=20)
        layout = QGridLayout()
        cbox = gui.widgetBox(self.controlArea, "Save advanced", margin=10, orientation=layout, addSpace=True, stretch=1)
        c1 = gui.checkBox(None, self, 'enable_multiselect', 'Enable multi selection', callback=self.check_multiselect)
        layout.addWidget(c1, 0, 0, 1, 3)

        li = gui.label(None, self, 'Input data', addSpace=True, )
        li.setStyleSheet('color: green')
        layout.addWidget(li, 2, 0, 1, 3)

        self.tvInput = DictionaryTreeWidget(self, None, editable=True)
        layout.addWidget(self.tvInput, 3, 0, 1, 3)

        vbox = gui.vBox(None, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.button(vbox, self, 'All\n==>', callback=self.sendOutputAll, autoDefault=False,
                   sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.button(vbox, self, 'Selected\n==>', callback=self.sendOutput, autoDefault=False,
                   sizePolicy=(Policy.Fixed, Policy.Fixed))
        layout.addWidget(vbox, 3, 3)

        ls = gui.label(None, self, 'Load file: ')
        layout.addWidget(ls, 0, 4)

        self.file_combo.setMinimumWidth(150)
        self.file_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.file_combo.activated[int].connect(self.select_file)
        layout.addWidget(self.file_combo, 0, 5, 1, 3)

        f1 = gui.button(None, self, 'Load', autoDefault=False, stretch=1, callback=self.browse_file,
                        sizePolicy=(Policy.Fixed, Policy.Fixed), icon=self.style().standardIcon(QStyle.SP_FileIcon))
        f2 = gui.button(None, self, 'Reload', autoDefault=False, stretch=1, callback=lambda: self.load_file(),
                        sizePolicy=(Policy.Fixed, Policy.Fixed),
                        icon=self.style().standardIcon(QStyle.SP_BrowserReload))
        f3 = gui.button(None, self, 'Clear', autoDefault=False, stretch=1, callback=self.clear_file,
                        sizePolicy=(Policy.Fixed, Policy.Fixed),
                        icon=self.style().standardIcon(QStyle.SP_DialogCloseButton))
        layout.addWidget(f1, 1, 5)
        layout.addWidget(f2, 1, 6)
        layout.addWidget(f3, 1, 7)

        self.lblFile = gui.label(None, self, 'No save file selected')
        self.lblFile.setStyleSheet('color: green')
        layout.addWidget(self.lblFile, 2, 4, 1, 4)

        self.tvOutput = DictionaryTreeWidget(self, None, editable=True, enable_dragdrop=True)
        layout.addWidget(self.tvOutput, 3, 4, 1, 4)

        self.btnCancel = gui.button(None, self, 'Cancel changes', callback=lambda: self.load_file(), autoDefault=False,
                                    stretch=1, sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.btnSave = gui.button(None, self, 'Save ', callback=lambda: self.save_changes(overwrite=False),
                                  autoDefault=False, stretch=1, sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.btnSaveOW = gui.button(None, self, 'Save (overwrite file)',
                                    callback=lambda: self.save_changes(overwrite=True), autoDefault=False,
                                    stretch=1, sizePolicy=(Policy.Fixed, Policy.Fixed))

        layout.addWidget(self.btnCancel, 4, 5)
        layout.addWidget(self.btnSave, 4, 6)
        layout.addWidget(self.btnSaveOW, 4, 7)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(topbox)
        splitter.addWidget(cbox)
        self.controlArea.layout().addWidget(splitter)

        # self.tvOutput.setEnabled(False)
        self.tvOutput.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tvOutput.customContextMenuRequested.connect(self.rightClickTreeItem)
        self.menuMain = QtWidgets.QMenu()
        new_grp = self.menuMain.addAction('Add new group')
        new_grp.triggered.connect(self.create_new_group)
        note = self.menuMain.addAction('Add note')
        note.triggered.connect(self.create_note)

        self.menuGroup = QtWidgets.QMenu()
        sub_grp = self.menuGroup.addAction("Add new sub group")
        sub_grp.triggered.connect(self.create_new_group)
        clr_grp = self.menuGroup.addAction("Clear all group items")
        clr_grp.triggered.connect(self.clear_group)
        del_grp = self.menuGroup.addAction("Delete group")
        del_grp.triggered.connect(self.del_item)

        self.menuData = QtWidgets.QMenu()
        del_data = self.menuData.addAction("Delete dataset")
        del_data.triggered.connect(self.del_item)

    def sizeHint(self):
        return QSize(900, 600)

    def _clean_recentpaths(self):
        pathlist = []
        for i, item in enumerate(self.recent_paths):
            if i > 20:
                break
            if Path(item.abspath).exists():
                pathlist.append(item)
        self.recent_paths = pathlist

    @Inputs.data
    def set_data(self, data, id):
        self.clear_messages()
        # self.load_title(data, id)
        self.id_name_map[id] = f'_{id[0]}'
        if id in self.id_name_map:
            name = self.id_name_map[id]
            if name in self.data_in:
                if data is None:
                    del self.data_in[name]
                else:
                    self.data_in[name] = data
            else:
                if data is not None:
                    self.data_in[name] = data

        self.data_in_sel = list(self.data_in.values())[0] if len(self.data_in) == 1 else self.data_in

        if len(self.data_in) > 0:
            self.load_data()
        else:
            self.no_input()
            # self.btnCancel.setEnabled(False)
            # self.btnSave.setEnabled(False)

    def load_title(self, data, id):
        # if not any(data):
        #     return
        if id not in self.id_name_map.keys():
            # if 'filename' in data and data['filename']!='' and data['filename'] not in list(self.id_name_map.values()):
            #     self.id_name_map[id] = '{}'.format(data['filename'])
            # else:
            text, ok = QInputDialog.getText(self, 'Data name ({})'.format(self.name),
                                            'Enter name for the saving datasets:')
            self.id_name_map[id] = str(text) if ok and str(text) != '' else '_{}'.format(id[0])

    def load_data(self):
        super().load_data(multi=True)
        # self.btnCancel.setEnabled(True)
        # self.btnSave.setEnabled(True)
        self.tvInput.updateDictionary(self.data_in_sel)
        if self.save_updates and os.path.exists(self.last_path()):
            self.save_all(load_new_file=False)

    def select_file(self, n):
        assert n < len(self.recent_paths)
        super().select_file(n)
        if self.recent_paths:
            self.load_file()
            self.set_file_list()

    def no_input(self):
        self.data_in = {}
        self.data_in_sel = {}
        # self.data_out = {}
        self.infoInput.setText('No input data')
        self.tvInput.updateDictionary({})
        # self.tvOutput.updateDictionary({})

    def del_item(self):
        if self.selitem is not None:
            items = self.tvOutput.get_selected_path(self.selitem, [])
            item_text = ''
            data = self.data_out
            for i, item in enumerate(items):
                item_text += '/{}'.format(item)
                if i == len(items) - 1:
                    del data[item]
                else:
                    data = data[item]
            self.tvOutput.updateDictionary(self.data_out)
            self.lblFile.setText('Deleted {}'.format(item_text))
            self.lblFile.setStyleSheet('color: red')

    def clear_group(self):
        if self.selitem is not None:
            data = self.tvOutput.get_selected_data(self.selitem)
            data.clear()
            self.tvOutput.updateDictionary(self.data_out)
            self.lblFile.setText('Cleared group {}'.format(self.selitem.text(0)))
            self.lblFile.setStyleSheet('color: red')

    def create_new_group(self):
        data = self.data_out
        if self.selitem is not None:
            data = self.tvOutput.get_selected_data(self.selitem)
        text, ok = QInputDialog.getText(self, self.name, 'Group name: ')
        if ok:
            if text in data:
                self.Error.unknown('Group name already exists')
            else:
                data[text] = {}
                self.tvOutput.updateDictionary(self.data_out)

    def create_note(self):
        note = str(self.data_out['note']) if 'note' in self.data_out else ''
        text, ok = QInputDialog.getMultiLineText(self, self.name, 'Note : ', note)
        if ok:
            self.data_out['note'] = text
            self.tvOutput.updateDictionary(self.data_out)

    def rightClickTreeItem(self, point):
        # if self.reader is not None:
        self.selitem = self.tvOutput.itemAt(point)
        if self.selitem is None:
            self.menuMain.exec_(self.tvOutput.mapToGlobal(point))
        else:
            self.selData = self.tvOutput.get_selected_data(baseNode=self.selitem)
            if isinstance(self.selData, dict):
                self.menuGroup.exec_(self.tvOutput.mapToGlobal(point))
            else:
                self.menuData.exec_(self.tvOutput.mapToGlobal(point))

    def browse_file(self):
        start_file = self.last_path() or os.path.expanduser("~/")
        if len(self.data_in) > 0:
            start_file = list(self.data_in.values())[0].get('comment_log', start_file)
            pathidx = start_file.find('Data loaded from file ')
            if pathidx != -1:
                start_file = start_file.split('Data loaded from file ')[-1][:-1]
                start_file = str(Path(start_file).parent)
                if 'sequence' in start_file:
                    start_file = start_file[10:]
        filt = ['.pkl', '.h5', '.dat', '.txt']
        readers = [f for f in FileFormat.formats
                   if getattr(f, 'read', None)
                   and getattr(f, "EXTENSIONS", None)
                   and any(set(getattr(f, "EXTENSIONS", None)).intersection(filt))]
        filename, reader, _ = open_filename_dialog_save(start_file, None, readers)
        if filename is not None:
            self.add_path(filename)
            if reader is not None:
                self.recent_paths[0].file_format = reader.qualified_name()
                # if 'h5' in filename.rpartition('.')[-1] and len(self.data_in) > 0:
                #     id, data = list(self.data_in.items())[0]
                #     self.load_title(data, id)
            if os.path.exists(filename):
                self.load_file()
            else:
                self.lblFile.setText('Creating new file {}'.format(os.path.basename(self.last_path())))
                self.lblFile.setStyleSheet('color: blue')
                self.data_out = {}
                self.tvOutput.updateDictionary(self.data_out)
                # self.tvOutput.setEnabled(True)
                try:
                    self.reader = self._get_reader()
                    assert self.reader is not None
                except Exception:
                    self.lblFile.setText('File reader not found')
                    return self.Error.missing_reader
        return filename

    def _get_reader(self) -> FileFormat:
        path = self.last_path()
        if path is None:
            return None
        if self.recent_paths and self.recent_paths[0].file_format:
            qname = self.recent_paths[0].file_format
            reader_class = class_from_qualified_name(qname)
            reader = reader_class(path)
        else:
            reader = FileFormat.get_reader(path)
        if self.recent_paths and self.recent_paths[0].sheet:
            reader.select_sheet(self.recent_paths[0].sheet)
        return reader

    def load_file(self, open_dialog=True):
        self.load_flag = False
        self.setStatusMessage('')
        self.clear_messages()
        self.set_file_list()

        if self.last_path() and not os.path.exists(self.last_path()):
            self.lblFile.setText('File not found')
            return self.Error.file_not_found

        try:
            self.reader = self._get_reader()
            assert self.reader is not None
        except Exception:
            self.lblFile.setText('File reader not found')
            return self.Error.missing_reader

        try:
            if hasattr(self.reader, 'open_dialog'):
                setattr(self.reader, 'open_dialog', open_dialog)
            if hasattr(self.reader, 'selected_group') and self.selected_group != '':
                setattr(self.reader, 'selected_group', self.selected_group)
            data = self.reader.read()
            self.selected_group = getattr(self.reader, 'selected_group', '')
            if type(data) is dict:
                self.data_out = data
                self.tvOutput.updateDictionary(self.data_out)
                # self.tvOutput.setEnabled(True)
                if np.any(data):
                    self.lblFile.setText('Loaded existing file {}'.format(os.path.basename(self.last_path())))
                    self.lblFile.setStyleSheet('color: orange')
                else:
                    self.lblFile.setText('File is empty')
                    self.lblFile.setStyleSheet('color: blue')
                if self.selected_group != '':
                    self.lblFile.setText(
                        'Loaded existing file {}: group = {}'.format(os.path.basename(self.last_path()),
                                                                     self.selected_group))
                self.load_flag = True
            else:
                self.lblFile.setText('Please implement File reader in dictionary format')
                self.lblFile.setStyleSheet('color: red')
        except Exception as ex:
            self.data_out = {}
            self.tvOutput.updateDictionary(self.data_out)
            self.lblFile.setText('Exception reading file')
            return self.Error.unknown(str(ex))

    def clear_file(self):
        if self.last_path() is not None:
            self.data_out = {}
            self.tvOutput.updateDictionary(self.data_out)
            self.lblFile.setText('Cleared file data')
            self.lblFile.setStyleSheet('color: red')

    def save_changes(self, overwrite=True, reload=True):
        try:
            self.clear_messages()
            path = self.last_path()
            if path is None or not self.load_flag:
                tmp = {}
                copy_items(self.data_out, tmp)
                if path is None:
                    path = self.browse_file()
                copy_items(tmp, self.data_out)
                if path is None:
                    self.info.set_output_summary('Saving cancelled')
                    self.lblFile.setText('Saving cancelled')
                    self.lblFile.setStyleSheet('color: red')
                    self.tvOutput.clear()
                    return
            if self.reader is not None:
                group_cmt = ''
                if self.selected_group != '' and hasattr(self.reader, 'selected_group'):
                    setattr(type(self.reader), 'selected_group', self.selected_group)
                    group_cmt = ', group = {}'.format(self.selected_group)
                if hasattr(self.reader, 'overwrite'):
                    setattr(type(self.reader), 'overwrite', overwrite)
                self.reader.write_file(path, self.data_out)
                if reload and questionMsg(self, 'Reload', 'Saved successfully. Load / reload file?'):
                    self.load_file(open_dialog=False)
                else:
                    self.load_flag = False
                    self.data_out = {}
                    self.tvOutput.clear()
                    self.add_path(os.path.expanduser("~/"))
                self.info.set_output_summary('Saved to file: {}{}'.format(os.path.basename(path), group_cmt))
                self.setStatusMessage('Saved to file: {}{}'.format(os.path.basename(path), group_cmt))
                self.lblFile.setText('Saved updates to file {}{}'.format(os.path.basename(path), group_cmt))
                self.lblFile.setStyleSheet('color: green')
                # Reset class variables to default (for HDF5reader)
                if self.selected_group != '' and hasattr(self.reader, 'selected_group'):
                    setattr(type(self.reader), 'selected_group', '')
                if hasattr(self.reader, 'overwrite'):
                    setattr(type(self.reader), 'overwrite', True)
            else:
                self.info.set_output_summary('File not loaded/ reader not found')
                self.setStatusMessage('')
        except Exception as e:
            self.Error.unknown(repr(e))

    def save_all(self, obj=False, load_new_file=True):
        if np.any(self.data_in_sel):
            if load_new_file:
                self.browse_file()
            self.data_out = {}
            copy_items(self.data_in_sel, self.data_out, deepcopy=True)
            self.save_changes(reload=False)
        else:
            raise Exception('No input data.')

    def sendOutputAll(self):
        self.clear_messages()
        # if self.last_path() is not None:
        if any(self.data_out):
            val = questionMsgAdv(title='Merge data?',
                                 msg='File has data. Press "yes" to update file and  "no" to replace file')
            if val == 1:
                self.data_out = {}
            elif val == 0:
                return
        copy_items(self.data_in_sel, self.data_out, deepcopy=True)
        self.tvOutput.updateDictionary(self.data_out)
        if self.load_flag:
            self.info.set_output_summary('Modified file')
            self.lblFile.setText('Modified file. Click "Save" button for updating file.')
            self.lblFile.setStyleSheet('color: red')
        else:
            self.info.set_output_summary('Updated output channel')
            self.lblFile.setText('Data in output channel. Click "Save" button for saving to a file.')
            self.lblFile.setStyleSheet('color: red')

    def sendOutput(self):
        self.clear_messages()
        # if self.last_path() is not None:
        selected = self.tvInput.selectedItems()
        if any(selected):
            for item in selected:
                data = self.tvInput.get_selected_data(baseNode=item)
                key = item.text(0)
                self.data_out[key] = data

            self.tvOutput.updateDictionary(self.data_out)
            if self.load_flag:
                self.info.set_output_summary('Modified file')
                self.lblFile.setText('Modified file. Click "Save" button for updating file.')
                self.lblFile.setStyleSheet('color: red')
            else:
                self.info.set_output_summary('Updated output channel')
                self.lblFile.setText('Data in output channel. Click "Save" button for saving to a file.')
                self.lblFile.setStyleSheet('color: red')

    def check_multiselect(self):
        if self.enable_multiselect:
            self.tvInput.setSelectionMode(QAbstractItemView.MultiSelection)
        else:
            self.tvInput.setSelectionMode(QAbstractItemView.SingleSelection)
