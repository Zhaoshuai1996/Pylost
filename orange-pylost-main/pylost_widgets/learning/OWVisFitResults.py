# coding=utf-8
"""OWFileExternal is copied from orange native OWFile and adapted to pylost use
"""

import numpy as np
from Orange.data import Table
from Orange.data.variable import ContinuousVariable
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QSizePolicy
from orangewidget.settings import Setting
from orangewidget.widget import Msg
from silx.gui.plot import Plot1D

from pylost_widgets.util.util_plots import CurveLegendListContextMenu

DEG_TO_MRAD = 17.4533


class OWVisFitResults(widget.OWWidget):
    name = 'Visualize regression'
    description = 'Visualize results of different regression fits.'
    icon = "../icons/plot.svg"
    priority = 1001

    class Inputs:
        data = Input('fit_data', Table)

    want_main_area = False
    sel_x = Setting('')
    sel_y = Setting('')

    class Error(widget.OWWidget.Error):
        unknown = Msg("Error:\n{}")

    class NoFileSelected:
        pass

    def __init__(self):
        super().__init__()
        self.data_in = {}
        self.plot_arr = []

        box = gui.vBox(self.controlArea, "Info", stretch=1)
        self.infolabel = gui.widgetLabel(box, 'No data loaded.')

        box = gui.hBox(self.controlArea, "New", stretch=1)
        self.selX = gui.comboBox(box, self, "sel_x", label='X-axis:', labelWidth=50, sendSelectedValue=True,
                                 orientation=Qt.Horizontal, stretch=1,
                                 sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.selY = gui.comboBox(box, self, "sel_y", label='Y-axis:', labelWidth=50, sendSelectedValue=True,
                                 orientation=Qt.Horizontal, stretch=1,
                                 sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.btnAdd = gui.button(box, self, 'Add plot', callback=self.addPlot, autoDefault=False, stretch=1,
                                 sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.btnDiff = gui.button(box, self, 'Plot difference', callback=self.addPlotDiff, autoDefault=False, stretch=1,
                                  sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.btnClear = gui.button(box, self, 'Clear plot', callback=self.clearPlot, autoDefault=False, stretch=1,
                                   sizePolicy=QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        self.plot = Plot1D()
        box = gui.vBox(self.controlArea, "Info", stretch=18)
        box.layout().addWidget(self.plot)
        try:
            ld_list = self.plot.getLegendsDockWidget()._legendWidget
            contextMenu = CurveLegendListContextMenu(ld_list.model(), self.plot)
            contextMenu.sigContextMenu.connect(ld_list._contextMenuSlot)
            ld_list.setContextMenu(contextMenu)
        except Exception as e:
            self.Error.unknown(repr(e))

    def sizeHint(self):
        return QSize(1000, 700)

    @Inputs.data
    def set_data(self, data):
        if data is not None:
            self.data_in = data
            self.load_data()
        else:
            self.data_in = {}
            self.infolabel.setText('No data')

    def load_data(self):
        if np.any(self.data_in):
            domain = self.data_in.domain
            names = ['Select'] + [x.name for x in domain.variables]
            metas = [x.name for x in domain.metas if isinstance(x, ContinuousVariable)]
            self.infolabel.setText('Loaded table with {} rows'.format(len(self.data_in)))
            self.selX.clear()
            self.selY.clear()
            self.x_items = names
            self.y_items = names + metas
            self.selX.addItems(self.x_items)
            self.selY.addItems(self.y_items)
            # self.clearPlot()
            self.plot.getLegendsDockWidget().setVisible(True)
            self.load_plots()

    def load_plots(self):
        if any(self.plot_arr):
            for ixy in self.plot_arr:
                if ixy[0] in self.x_items and ixy[1] in self.y_items:
                    x, y, idx = self.get_xy(ixy[0], ixy[1])
                    if ixy[2] == 'a':
                        self.plot.addCurve(x[idx], y[idx], legend='{} - {}'.format(ixy[1], ixy[0]))
                    elif ixy[2] == 'd':
                        self.plot.addCurve(x[idx], y[idx] - x[idx], legend='Difference {} - {}'.format(ixy[1], ixy[0]))

    def addPlot(self):
        self.plot_arr.append((self.sel_x, self.sel_y, 'a'))
        x, y, idx = self.get_xy()
        self.plot.addCurve(x[idx], y[idx], legend='{} - {}'.format(self.sel_x, self.sel_y))

    def addPlotDiff(self):
        self.plot_arr.append((self.sel_x, self.sel_y, 'd'))
        x, y, idx = self.get_xy()
        self.plot.addCurve(x[idx], y[idx] - x[idx], legend='Difference {} - {}'.format(self.sel_y, self.sel_x))

    def get_xy(self, ix=None, iy=None):
        if ix is None:
            ix = self.sel_x
        if iy is None:
            iy = self.sel_y
        x = np.array(self.data_in[:, ix]).ravel()
        y = np.array(self.data_in[:, iy]).ravel()
        if not np.any(y):
            y = np.array(self.data_in[:, iy].metas).ravel()
        x_idx = np.argsort(x)
        return x, y, x_idx

    def clearPlot(self):
        self.plot.remove()
        self.plot_arr = []
