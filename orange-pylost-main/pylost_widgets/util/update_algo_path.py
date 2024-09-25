# coding=utf-8
import datetime
import importlib
import inspect
import os
import pkgutil

import numpy as np
from PyQt5 import uic
from PyQt5.QtCore import QMetaObject
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QListWidgetItem

from PyLOSt.databases.gs_table_classes import AlgoTypes, Algorithms, ConfigParams, InputDispTypes, Locations, \
    StitchSetupAlgoOptions
from pylost_widgets.util.resource_path import resource_path
from pylost_widgets.util.util_functions import questionMsg

qtCreatorFile = resource_path(os.path.join("gui", "dialog_update_algopath.ui"))  # Enter file here.
UI_import, QtBaseClass = uic.loadUiType(qtCreatorFile)


class UpdateAlgoPath(QDialog, UI_import):
    def __init__(self, parent=None, algo_type='S'):
        """
        Update algorithms paths (directories). Default path 'PyLOSt.algorithms.stitching' is used
        for default stitching algorithms present within PyLOST. New paths can be added via StitchParams widget.

        :param parent: Parent object
        :type parent: QWidget
        :param algo_type: Type of algorithm: 'S' = Stitching
        :type algo_type: str
        """
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.sel_item_id = -1
        self.algo_type = algo_type
        if self.algo_type == 'S':
            self.param_name = 'ALGO_LOC_STITCH'
            self.param_val_default = 'PyLOSt.algorithms.stitching'

        self.load_list()
        self.listWidget.itemClicked.connect(self.list_click)
        self.add_path.clicked.connect(self.add_click)
        self.remove_path.clicked.connect(self.remove_click)
        self.reload.clicked.connect(self.reload_click)
        self.buttonBox.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        self.buttonBox.button(QDialogButtonBox.Reset).clicked.connect(self.reset_selection)

    def restore_defaults(self):
        """
        Restore default paths.
        """
        self.delete_paths(del_all=True)
        self.add_default()
        self.load_list()

    def delete_paths(self, del_all=False):
        """
        Delete first or all paths.

        :param del_all: Flag to delete all paths
        :type del_all: bool
        """
        try:
            qpaths = ConfigParams.selectBy(paramName=self.param_name)
            if del_all:
                for qp in qpaths:
                    qp.destroySelf()
            else:
                qpaths[0].destroySelf()
        except Exception as e:
            print(e)

    def add_default(self):
        """Add default path, e.g. for stitching algorithms default path is 'PyLOSt.algorithms.stitching' """
        self.add_new(self.param_name, 'Default package with stitching algorithms (orange-pylost)',
                     self.param_val_default)

    @staticmethod
    def add_new(name, description, value, type='D'):
        """
        Add new path to the algorithm files.

        :param name: Configuration parameter name
        :type name: str
        :param description: Description of the new path
        :type description: str
        :param value: New path
        :type value: str
        :param type: Type of config parameter, 'D' = Discrete, 'C' = Continuous
        :type type: str
        """
        try:
            ConfigParams(paramName=name, paramDesc=description, paramType=type, paramValue=value,
                         dateCreated=datetime.datetime.today().strftime('%Y-%m-%d'))
        except Exception as e:
            print(e)

    def load_list(self):
        """
        Load list of algorithm directory paths.
        """
        qpaths = ConfigParams.selectBy(paramName=self.param_name)
        self.listWidget.clear()
        for it in qpaths:
            item = QListWidgetItem()
            item.setData(0, it.paramValue)
            item.setData(1, it.id)
            self.listWidget.addItem(item)

    def list_click(self, item):
        """
        Click callback of the displayed list of paths.

        :param item: Clicked item
        :type item: QListWidgetItem
        """
        self.sel_item_id = item.data(1)

    def reset_selection(self):
        """
        Reset selection the list viewer
        """
        self.sel_item_id = -1
        self.listWidget.setCurrentItem(None)

    def remove_click(self):
        """
        Remove selected path
        """
        try:
            if self.sel_item_id != -1:
                ConfigParams.selectBy(id=self.sel_item_id)[0].destroySelf()
                self.load_list()
        except Exception as e:
            print(e)

    def add_click(self):
        """
        Load and save a new algorithm directory path
        """
        try:
            fdir = str(QFileDialog.getExistingDirectory(self, "Select algorithm files directory"))
            if fdir:
                self.add_new(self.param_name, 'Additional package with stitching algorithms (orange-pylost)', fdir)
                self.load_list()
        except Exception as e:
            print(e)

    def reload_click(self):
        """
        Save changes and reload list viewer containing paths
        """
        pkg_paths = []
        del_all = False
        if self.sel_item_id == -1:
            if questionMsg(title='Search full path', msg='No path is selected. Update algorithms in the full path? '
                                                         '\n (This procedure will completely erase any previous database information about algorithms)'):
                del_all = False
                qpaths = ConfigParams.selectBy(paramName=self.param_name)
                for qp in qpaths:
                    path = qp.paramValue
                    if os.path.exists(path):
                        pkg_paths.append(path)
                    else:
                        pkg = importlib.import_module(path, __package__)
                        pkg_paths.append(os.path.dirname(pkg.__file__))
        else:
            path = self.listWidget.currentItem().text()
            if os.path.exists(path):
                pkg_paths.append(path)
            else:
                pkg = importlib.import_module(path, __package__)
                pkg_paths.append(os.path.dirname(pkg.__file__))

        for (module_loader, module_name, ispkg) in pkgutil.walk_packages(pkg_paths):
            if not ispkg:
                module = module_loader.find_module(module_name).load_module(module_name)
                # print('module={}'.format(module))
                for class_name, obj in inspect.getmembers(module):
                    if isinstance(obj, type):
                        if hasattr(obj, 'name') and obj.name not in (None, ''):
                            print(class_name, type(obj), obj.name)
                            if Algorithms.selectBy(algoName=obj.name).count() > 0:
                                self.del_algo(obj.name, del_all)
                            self.add_algo(obj, obj.name, class_name)
            else:
                module_loader.find_module(module_name).load_module(module_name)

    @staticmethod
    def del_algo(name, del_all=False):
        """
        Delete a (stitching) algorithm.

        :param name: Name of algorithm
        :type name: str
        :param del_all: Delete all algorithms
        :type del_all: bool
        """
        if del_all:
            qalgos = Algorithms.select()
            for qa in qalgos:
                qa.destroySelf()
            qalgoopts = StitchSetupAlgoOptions.select()
            for qopt in qalgoopts:
                qopt.destroySelf()
        else:
            qalgo = Algorithms.selectBy(algoName=name)[0]
            if StitchSetupAlgoOptions.selectBy(algoID=qalgo.id).count() > 0:
                qalgoopts = StitchSetupAlgoOptions.selectBy(algoID=qalgo.id)
                for qopt in qalgoopts:
                    qopt.destroySelf()
            qalgo.destroySelf()

    @staticmethod
    def add_algo(obj, name, class_name):
        """
        Add a (stitching) algorithm

        :param obj: Algorithm object
        :type obj: Algorithms
        :param name: Name of algorithm
        :type name: str
        :param class_name: Name of the class implementing this algorithm
        :type class_name: str
        """
        ctype = None
        loc = None
        if AlgoTypes.selectBy(algoType=obj.ctype).count() > 0:
            qalgotype = AlgoTypes.selectBy(algoType=obj.ctype)[0]
            ctype = qalgotype.id
        if Locations.selectBy(location=obj.location).count() > 0:
            qloc = Locations.selectBy(location=obj.location)[0]
            loc = qloc.id
        Algorithms(algoName=name,
                   algoDesc=obj.description,
                   algoType=ctype,
                   functionName=class_name,
                   addedBy=obj.added_by,
                   location=loc,
                   dateAdded=datetime.datetime.today().strftime('%Y-%m-%d'))
        qalgo = Algorithms.selectBy(algoName=name)[0]

        # Add algo options
        attributes = [attr for attr in dir(obj) if
                      not callable(getattr(obj, attr)) and not attr.startswith('_') and attr not in ['name',
                                                                                                     'description',
                                                                                                     'ctype',
                                                                                                     'added_by',
                                                                                                     'location']]
        for attr in attributes:
            attr_default = getattr(obj, attr, None)
            if type(attr_default) is not dict:
                if isinstance(attr_default, QMetaObject):
                    continue
                attr_default = {'value': attr_default}

            attr_val = attr_default.get('value', '')
            def_val = str(attr_val[0] if type(attr_val) is tuple else attr_val)

            description = attr_default.get('description', '')
            disp_type = attr_default.get('disp_type', None)
            disp_type_id = None
            if disp_type not in (None, '') and InputDispTypes.selectBy(dispType=disp_type).count() > 0:
                qdisp = InputDispTypes.selectBy(dispType=disp_type)[0]
                disp_type_id = qdisp.id
            all_values = attr_default.get('all_values', None)
            if isinstance(all_values, (tuple, list, np.ndarray)):
                all_values = ','.join(str(x) for x in all_values)
            else:
                all_values = str(all_values)
            unit = attr_default.get('unit', None)
            group_items = attr_default.get('group_items', None)
            StitchSetupAlgoOptions(algoID=qalgo.id, option=attr, optionDesc=description,
                                   dispTypeID=disp_type_id, defVal=def_val, allVals=all_values,
                                   valUnit=unit, addedBy=obj.added_by, groupItems=group_items,
                                   dateAdded=datetime.datetime.today().strftime('%Y-%m-%d'))

    @staticmethod
    def update_algo(obj, name, class_name):
        """
        Update selected (stithcing) algorithm parameters.

        :param obj: Algorithm object
        :type obj: Algorithms
        :param name: Name of algorithm
        :type name: str
        :param class_name: Name of the class implementing this algorithm
        :type class_name: str
        """
        qalgo = Algorithms.selectBy(algoName=name)[0]
        qalgo.functionName = class_name
        if hasattr(obj, 'description') and obj.description not in (None, ''):
            qalgo.algoDesc = obj.description
        if hasattr(obj, 'ctype') and obj.ctype not in (None, ''):
            if AlgoTypes.selectBy(algoType=obj.ctype).count() > 0:
                qalgotype = AlgoTypes.selectBy(algoType=obj.ctype)[0]
                qalgo.algoType = qalgotype.id
        if hasattr(obj, 'added_by') and obj.added_by not in (None, ''):
            qalgo.addedBy = obj.added_by
        if hasattr(obj, 'location') and obj.location not in (None, ''):
            if Locations.selectBy(location=obj.location).count() > 0:
                qloc = Locations.selectBy(location=obj.location)[0]
                qalgo.location = qloc.id
