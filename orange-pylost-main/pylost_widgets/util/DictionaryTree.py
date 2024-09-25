# coding=utf-8
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QDialogButtonBox, QGridLayout, QTreeWidget, QTreeWidgetItem
from astropy.units import Quantity
from numpy.compat import unicode


class DictionaryTreeDialog(QDialog):
    def __init__(self, parent=None, data=None, title='Select item'):
        """
        Dialog to display tree structure of any python dictionary object.

        :param parent: Parent object
        :type parent: QWidget
        :param data: Dictionary data
        :type data: dict
        """
        QDialog.__init__(self, parent)
        self.data_sel = None
        self.item_sel = None
        self.data = data
        layout = QGridLayout()
        self.setLayout(layout)
        self.setWindowTitle(title)
        self.dictTree = DictionaryTreeWidget(self, data)
        self.dictTree.clicked.connect(self.selectItem)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.dictTree)
        layout.addWidget(self.buttonBox)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def selectItem(self):
        """
        Select data of clicked item.
        """
        self.item_sel = self.dictTree.currentItem()
        self.data_sel = self.dictTree.get_selected_data()

    def get_selected_data(self):
        if self.item_sel is not None:
            if isinstance(self.data_sel, dict):
                return self.data_sel
            else:
                return {self.item_sel.text(0): self.data_sel}
        return self.data


class DictionaryTreeWidget(QTreeWidget):
    def __init__(self, parent=None, data=None, editable=False, enable_dragdrop=False):
        """
        Widget to display tree structure of any python dictionary object.

        :param parent: Parent object
        :type parent: QWidget
        :param data: Dictionary data
        :type data: dict
        :param editable: Flag if checked enables to edit dictionary keys
        :type editable: bool
        :param enable_dragdrop: Enable drag and drop
        :type enable_dragdrop: bool
        """
        QTreeWidget.__init__(self, parent)
        self.data = data
        self.editable = editable
        self.enable_dragdrop = enable_dragdrop
        self.setHeaderLabels(['Name', 'Type', 'Shape', 'Unit'])
        self.updateDictionary(data)
        self.prev_name = ''
        if self.editable:
            self.rename_flag = False
            self.itemDoubleClicked.connect(self.double_click)
            self.itemChanged.connect(self.edit_item)
        if enable_dragdrop:
            self.setSelectionMode(self.ExtendedSelection)
            self.setDragDropMode(self.InternalMove)
            self.setDragEnabled(True)
            self.setDropIndicatorShown(True)

    def updateDictionary(self, data):
        """
        Update displayed dictionary data.

        :param data: Dictionary data
        :type data: dict
        """
        self.data = data
        self.clear()
        if data is not None and type(data) is dict:
            self.fill_item(self.invisibleRootItem(), data)

    def fill_item(self, item, value):
        """
        Build tree by filling item by item.

        :param item: Parent Treewidget item
        :type item: QTreeWidgetItem
        :param value: Dictionary of values to add to item as children
        :type value: dict
        """
        # item.setExpanded(True)
        if type(value) is dict:
            for key, val in sorted(value.items()):
                child = QTreeWidgetItem()
                if self.editable:
                    child.setFlags(child.flags() | Qt.ItemIsEditable)
                if self.enable_dragdrop:
                    child.setFlags(child.flags() | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled)

                child.setText(0, unicode(key))
                if type(val) is dict:
                    item.setData(0, Qt.ForegroundRole, None)
                    child.setForeground(0, QBrush(QColor("green")))
                child.setText(1, type(val).__name__)
                if isinstance(val, np.ndarray):
                    child.setText(2, ' x '.join(str(dim) for dim in val.shape))
                elif isinstance(val, (list, tuple)):
                    child.setText(2, str(len(val)))
                elif np.isscalar(val):
                    child.setText(2, 'scalar')
                if isinstance(val, Quantity):
                    child.setText(3, '{}'.format(val.unit))

                item.addChild(child)
                self.fill_item(child, val)

    def get_selected_data(self, baseNode=None):
        """
        Get data of selected tree node.

        :param baseNode: Selected tree node
        :type baseNode: QTreeWidgetItem
        :return: Selected data
        :rtype: dict/object
        """
        if baseNode is None:
            baseNode = self.currentItem()
        items = self.get_selected_path(baseNode, [])
        data = self.data
        for item in items:
            data = data[item]
        return data

    def get_selected_path(self, baseNode, items):
        """
        Get path of selected tree node.

        :param baseNode: Selected tree node
        :type baseNode: QTreeWidgetItem
        :param items: List containing earlier path until current node
        :type items: list[str]
        :return: Selected path
        :rtype: list[str]
        """
        item = baseNode.text(0)
        items = [item] + items
        if baseNode.parent() is not None:
            items = self.get_selected_path(baseNode.parent(), items)
        return items

    def clearData(self):
        """
        Clear data of the tree widget
        """
        self.data = None
        self.clear()

    def double_click(self, item, column):
        """
        Callback of double click of a tree node. Used for renaming node.

        :param item: Clicked tree node
        :type item: QTreeWidgetItem
        :param column: Column position of item
        :type column: int
        """
        self.prev_name = item.text(column)
        self.rename_flag = True

    def edit_item(self, item, column):
        """
        Callback when tree item text is changed. Item renamed if used with double_click.

        :param item: Clicked tree node
        :type item: QTreeWidgetItem
        :param column: Column position of item
        :type column: int
        """
        if column == 0 and self.rename_flag and self.prev_name != item.text(column):
            self.update_dict_key(self.prev_name, item)
            self.rename_flag = False

    def update_dict_key(self, old_key, baseNode):
        """
        Update dictionary key with the new name entered with double click in the tree widget item.

        :param old_key: Old key in the dictionary
        :type old_key: str
        :param baseNode: QTreewidget item with the new name
        :type baseNode: QTreeWidgetItem
        """
        new_key = baseNode.text(0)
        path = self.get_selected_path(baseNode, [])
        data = self.data
        for k in path:
            if k != new_key:
                data = data[k]
            elif old_key in data.keys():  # Last item in loop
                if new_key != '':
                    data[k] = data.pop(old_key)
                else:
                    data.pop(old_key)
                    self.updateDictionary(self.data)

    def dropEvent(self, event):
        """Event in 'drag and drop' action, after dropped"""
        if event.source() == self:
            QAbstractItemView.dropEvent(self, event)

    def dropMimeData(self, parent, row, data, action):
        """
        Event when an item is moved.

        :param parent: Parent object
        :type parent: QWidget
        :param row: Row number of moving item
        :type row: int
        :param data: Data of the DictionaryTree
        :type data: dict
        :param action: Action object
        :type action: QAction
        :return: Flag whether item is moved
        :rtype: bool
        """
        if action == QtCore.Qt.MoveAction:
            return self.moveSelection(parent, row)
        return False

    ## https://riverbankcomputing.com/pipermail/pyqt/2009-December/025379.html
    def moveSelection(self, parent, position):
        """
        Drag and drop items within the dictionary tree widget.

        :param parent: Previous parent of the moving item
        :type parent: QTreeWidgetItem
        :param position: New row position to drop
        :type position: int
        :return: Flag whether item is moved
        :rtype: bool
        """
        # save the selected items
        selection = [QtCore.QPersistentModelIndex(i)
                     for i in self.selectedIndexes()]
        parent_index = self.indexFromItem(parent)
        if parent_index in selection:
            return False
        # save the drop location in case it gets moved
        target = self.model().index(position, 0, parent_index).row()
        if target < 0:
            target = position
        # remove the selected items
        taken = []
        taken_data = []
        for index in reversed(selection):
            item = self.itemFromIndex(QtCore.QModelIndex(index))
            data_item = self.data
            if item is None or item.parent() is None:
                topitem = self.takeTopLevelItem(index.row())
                taken.append(topitem)
                taken_data.append(data_item.pop(topitem.text(0)) if topitem is not None else None)
            else:
                item_path = self.get_selected_path(item, [])
                for k in item_path[:-1]:
                    data_item = data_item[k]
                taken.append(item.parent().takeChild(index.row()))
                taken_data.append(data_item.pop(item.text(0)))

        data = self.get_selected_data(parent) if parent_index.isValid() else self.data
        # insert the selected items at their new positions
        while taken:
            item = taken.pop(0)
            item_data = taken_data.pop(0)
            if item is not None and isinstance(item, QTreeWidgetItem):
                if isinstance(data, dict):
                    data[item.text(0)] = item_data
                else:
                    self.data[item.text(0)] = item_data
            if position == -1:
                # append the items if position not specified
                if parent_index.isValid() and isinstance(data, dict):
                    parent.insertChild(parent.childCount(), item)
                else:
                    self.insertTopLevelItem(self.topLevelItemCount(), item)
            else:
                # insert the items at the specified position
                if parent_index.isValid() and isinstance(data, dict):
                    parent.insertChild(min(target, parent.childCount()), item)
                else:
                    self.insertTopLevelItem(min(target, self.topLevelItemCount()), item)
        return True
