# coding=utf-8
import ast
import logging

import numpy as np
import pandas as pd
import silx
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QMenu
from silx.gui import qt
from silx.gui.plot import Plot1D, Plot2D, PlotWindow
from silx.gui.plot.LegendSelector import LegendListContextMenu, LegendsDockWidget
from silx.gui.plot.Profile import ProfileToolBar
from silx.utils.weakref import WeakMethodProxy

_logger = logging.getLogger(__name__)


class OrangePlot1D(Plot1D, PlotWindow):

    def __init__(self, parent=None, backend=None, show_legend=False):
        """
        Custom Plot1D implementation extending silx Plot1D.

        :param parent: Parent object
        :type parent: QWidget
        :param backend: Backend plotting agent 'matplotlib' or 'opengl'
        :type backend: str
        :param show_legend: Show legends widget
        :type show_legend: bool
        """
        super(OrangePlot1D, self).__init__(parent, backend)
        try:
            legend_list = self.getLegendsDockWidget()._legendWidget
            contextMenu = CurveLegendListContextMenu(legend_list.model())
            contextMenu.sigContextMenu.connect(legend_list._contextMenuSlot)
            legend_list.setContextMenu(contextMenu)
            if show_legend:
                self.getLegendsDockWidget().setVisible(True)
        except Exception as e:
            print(e)

    def getLegendsDockWidget(self):
        """
        Widget listing all the legends of the plot.

        :return: Legends dock widget
        :rtype: CurveLegendsDockWidget
        """
        if self._legendsDockWidget is None:
            self._legendsDockWidget = super().getLegendsDockWidget()
        if self._legendsDockWidget is not None:
            if type(self._legendsDockWidget) == LegendsDockWidget:
                self._legendsDockWidget.__class__ = CurveLegendsDockWidget
                self._legendsDockWidget.init()
        return self._legendsDockWidget


class OrangePlot2D(Plot2D, PlotWindow):

    def __init__(self, parent=None, backend=None, opacity=0.5, show_legend=False):
        """
        Custom Plot2D implementation extending silx Plot2D.

        :param parent: Parent object
        :type parent: QWidget
        :param backend: Backend plotting agent 'matplotlib' or 'opengl'
        :type backend: str
        :param opacity: Opacity of image
        :type opacity: float
        :param show_legend: Show legends widget
        :type show_legend: bool
        """
        # List of information to display at the bottom of the plot
        posInfo = [
            ('X', lambda x, y: x),
            ('Y', lambda x, y: y),
            ('Data', WeakMethodProxy(self._getImageValue)),
            ('Dims', WeakMethodProxy(self._getImageDims)),
        ]
        self.opacity = opacity
        super(OrangePlot2D, self).__init__(parent, backend)
        super(Plot2D, self).__init__(parent=parent, backend=backend,
                                     resetzoom=True, autoScale=False,
                                     logScale=False, grid=False,
                                     curveStyle=False, colormap=True,
                                     aspectRatio=True, yInverted=True,
                                     copy=True, save=True, print_=True,
                                     control=True, position=posInfo,
                                     roi=False, mask=True)
        if parent is None:
            self.setWindowTitle('Plot2D')
        self.getXAxis().setLabel('Columns')
        self.getYAxis().setLabel('Rows')
        self.setKeepDataAspectRatio(True)

        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == 'downward':
            self.getYAxis().setInverted(True)

        self.profile = ProfileToolBar(plot=self)
        self.addToolBar(self.profile)

        self.colorbarAction.setVisible(True)
        self.getColorBarWidget().setVisible(True)

        # Put colorbar action after colormap action
        actions = self.toolBar().actions()
        for action in actions:
            if action is self.getColormapAction():
                break

        try:
            legend_list = self.getLegendsDockWidget()._legendWidget
            contextMenu = ImageLegendListContextMenu(legend_list.model())
            contextMenu.sigContextMenu.connect(legend_list._contextMenuSlot)
            legend_list.setContextMenu(contextMenu)
            if show_legend:
                self.getLegendsDockWidget().setVisible(True)
        except Exception as e:
            print(e)

    def getLegendsDockWidget(self):
        """
        Widget listing all the legends of the plot.

        :return: Legends doct widget
        :rtype: ImageLegendsDockWidget
        """
        if self._legendsDockWidget is None:
            self._legendsDockWidget = super().getLegendsDockWidget()
        if self._legendsDockWidget is not None:
            if type(self._legendsDockWidget) == LegendsDockWidget:
                self._legendsDockWidget.__class__ = ImageLegendsDockWidget
                self._legendsDockWidget.init(alpha=self.opacity)
        return self._legendsDockWidget

    def hideImage(self, legend, flag=True):
        """Show/Hide the image associated to legend.
        Even when hidden, the image is kept in the list of images.

        :param legend: The legend associated to the image to be hidden
        :type legend: str
        :param flag: True (default) to hide the image, False to show it
        :type flag: bool
        """
        img = self._getItem('image', legend)
        if img is None:
            _logger.warning('Image not in plot: %s', legend)
            return

        isVisible = not flag
        if isVisible != img.isVisible():
            img.setVisible(isVisible)


class CurveLegendsDockWidget(LegendsDockWidget):
    sigCommonEvents = pyqtSignal(dict)

    def __init__(self, parent=None, plot=None):
        """
        Widget listing all the legends of the plot.
        Right click on a legend allows selection of further operations like offsetting or renaming the curve.

        :param parent: Parent object
        :type parent: QWidget
        :param plot: Plot widget
        :type plot: CustomPlot1DD
        """

        super().__init__(parent, plot)
        self.init()

    def init(self):
        """Initialize legends signal handler"""
        # self._legendWidget.sigLegendSignal.disconnect(super()._legendSignalHandler) # Keep the old signal handler
        self._legendWidget.sigLegendSignal.connect(self._legendSignalHandler)

    def _legendSignalHandler(self, ddict):
        """
        Handler for right click events. For some events common across all plot widgets, the event is forwarded.

        :param ddict: Event information dictionary
        :type ddict: dict
        """
        curve = None
        if 'legend' in ddict:
            curve = self.plot.getCurve(ddict['legend'])

        if ddict['event'] == "legendClicked":
            if ddict['button'] == "left":
                self.update_alpha(ddict['legend'])

        elif ddict['event'] == "setActiveCurve":
            self.update_alpha(ddict['legend'])

        elif ddict['event'] == "mouseClicked":
            self.update_alpha()

        elif ddict['event'] == "changeLineWidth":
            lwidth = curve.getLineWidth()
            text, ok = QInputDialog.getText(self, 'Line width', 'Enter new line width:', text='{}'.format(lwidth))
            if ok:
                lwidth = float(text)

        elif ddict['event'] == "changeColor":
            color = '{}'.format(curve.getColor())
            text, ok = QInputDialog.getText(self, 'Line width',
                                            'Enter new color, e.g. "black", "#6a4597", (red[0,1], green[0,1], blue[0,1], alpha[0,1]):',
                                            text=color)
            if ok:
                color = text
                if text.startswith('(') and text.endswith(')'):
                    try:
                        color = ast.literal_eval(text)
                    except Exception:
                        pass

        elif ddict['event'] == "offsetX":
            xoff = 0
            xabs = False
            text, ok = QInputDialog.getText(self, 'Offset X', 'Enter offset value:')
            if text == '':
                return
            if ok:
                xoff = float(text)
            else:
                text, ok = QInputDialog.getText(self, 'Offset X absolute', 'Enter absolute offset value:')
                if ok:
                    xabs = True
                    xoff = float(text)

        elif ddict['event'] == "offsetY":
            text, ok = QInputDialog.getText(self, 'Offset Y', 'Enter offset value:')
            if ok:
                self.offset_y(ddict['legend'], float(text))
            else:
                text, ok = QInputDialog.getText(self, 'Offset Y absolute', 'Enter absolute offset value:')
                if ok:
                    self.offset_y(ddict['legend'], float(text), absolute=True)

        elif ddict['event'] == "exportData":
            self.export_data()

        if ddict['event'] in ['removeCurve', 'offsetX', 'clearChanges', 'changeLineWidth', 'changeColor']:
            cdict = {
                'event': ddict['event'],
                'legend': ddict['legend']
            }
            if ddict['event'] == 'removeCurve':
                cdict['modelIndex'] = self._legendWidget.currentIndex()
            if ddict['event'] == "offsetX":
                cdict['offset'] = xoff
                cdict['absolute'] = xabs
            if ddict['event'] == "changeLineWidth":
                cdict['lineWidth'] = lwidth
            if ddict['event'] == "changeColor":
                cdict['color'] = color
            self.sigCommonEvents.emit(cdict)

    def renameCurve(self, oldLegend, newLegend):
        """
        Rename curve legend. The signal is forwarded to be implemented across all plot widgets

        :param oldLegend: Old legend
        :type oldLegend: str
        :param newLegend: New legend
        :type newLegend: str
        """
        super().renameCurve(oldLegend, newLegend)
        cdict = {
            'event': 'renameCurve',
            'legend': oldLegend,
            'newLegend': newLegend
        }
        self.sigCommonEvents.emit(cdict)

    def update_alpha_all(self, alpha=1):
        """
        Update alpha value for all curves

        :param alpha: Alpha value
        :type alpha: float
        """
        curves = self.plot.getAllCurves()
        for curve in curves:
            curve.setAlpha(alpha)

    def update_alpha(self, legend=''):
        """
        Update alpha of a curve.

        :param legend: Legend of the curve
        :type legend: str
        """
        curve = self.plot.getCurve(legend)
        if curve is not None:
            self.update_alpha_all(alpha=0.5)
            curve.setAlpha(1)
            curve.setHighlighted(False)
        else:
            self.update_alpha_all(alpha=1)

    def offset_x(self, legend, offset, absolute=False):
        """
        Offset curve along x axis.

        :param legend: Legend of the curve
        :type legend: str
        :param offset: Offset value
        :type offset: float
        :param absolute: Absolute or relative offset
        :type absolute: bool
        """
        try:
            curve = self.plot.getCurve(legend)
            if curve is not None:
                x, y, xerr, yerr = curve.getData()
                x = x - x[0] + offset if absolute else x + offset
                curve.setData(x, y, xerr, yerr)
                # self.plot.resetZoom()
        except Exception as e:
            print(e)

    def offset_y(self, legend, offset, absolute=False):
        """
        Offset curve along y axis.

        :param legend: Legend of the curve
        :type legend: str
        :param offset: Offset value
        :type offset: float
        :param absolute: Absolute or relative offset
        :type absolute: bool
        """
        try:
            curve = self.plot.getCurve(legend)
            if curve is not None:
                x, y, xerr, yerr = curve.getData()
                y = y - y[0] + offset if absolute else y + offset
                curve.setData(x, y, xerr, yerr)
                # self.plot.resetZoom()
        except Exception as e:
            print(e)

    def export_data(self):
        """
        Export data of all curves to excel / csv / txt files
        """
        try:
            out = pd.DataFrame()
            model = self._legendWidget.model()
            for i in range(model.rowCount()):
                if model.index(i).data(qt.Qt.CheckStateRole):
                    legend = str(model.index(i).data(Qt.DisplayRole))
                    curve = self.plot.getCurve(legend)
                    if curve is not None:
                        x, y, xerr, yerr = curve.getData()
                        out['{} {}'.format(legend, curve.getXLabel())] = x
                        out['{} {}'.format(legend, curve.getYLabel())] = y
            if any(out):
                filters = ['Microsoft Excel spreadsheet (*.xlsx)', 'Comma seperated values (*.csv)',
                           'Tab seperated text file (*.txt)']
                name, ext = QFileDialog.getSaveFileName(None, 'Save to file', '', ';;'.join(filters))
                if ext == filters[0]:
                    out.to_excel(name, sheet_name='Selected curves')
                elif ext == filters[1]:
                    out.to_csv(name)
                elif ext == filters[2]:
                    out.to_csv(name, sep='\t')
        except Exception as e:
            print(e)


class ImageLegendsDockWidget(LegendsDockWidget):
    sigCommonEvents = pyqtSignal(dict)

    def __init__(self, parent=None, plot=None):
        """
        Widget listing all the legends of the plot.
        Right click on a legend allows selection of further operations like offsetting or renaming the image.

        :param parent: Parent object
        :type parent: QWidget
        :param plot: Plot widget
        :type plot: CustomPlot2DD
        """
        super().__init__(parent, plot)
        self.init()

    def init(self, alpha=0.5):
        """Initialize legends signal handler"""
        self.alpha = alpha
        self._legendWidget.sigLegendSignal.disconnect(super()._legendSignalHandler)
        self._legendWidget.sigLegendSignal.connect(self._legendSignalHandler)

    def _legendSignalHandler(self, ddict):
        """
        Handler for right click events. For some events common across all plot widgets, the event is forwarded.

        :param ddict: Event information dictionary
        :type ddict: dict
        """
        _logger.debug("Legend signal ddict = %s", str(ddict))

        if ddict['event'] == "legendClicked":
            if ddict['button'] == "left":
                self.plot.setActiveImage(ddict['legend'])
                self.update_alpha(ddict['legend'])

        elif ddict['event'] == "removeCurve":
            self.plot.removeImage(ddict['legend'])

        elif ddict['event'] == "renameCurve":
            imageLegends = self.plot.getAllImages(just_legend=True)
            oldLegend = ddict['legend']
            newLegend, ok = QInputDialog.getText(self, 'Rename image', 'Image legend:', text=oldLegend)
            if ok:
                if newLegend in imageLegends:
                    raise Exception('Image name already exists')
                # self.renameImage(oldLegend, newLegend)

        elif ddict['event'] == "setActiveCurve":
            self.plot.setActiveImage(ddict['legend'])
            self.update_alpha(ddict['legend'])

        elif ddict['event'] == "checkBoxClicked":
            self.plot.hideImage(ddict['legend'], not ddict['selected'])

        elif ddict['event'] == "mouseClicked":
            self.update_alpha()

        elif ddict['event'] == "offsetX":
            xoff = 0
            xabs = False
            text, ok = QInputDialog.getDouble(self, 'Offset X', 'Enter offset value:')
            if ok:
                xoff = float(text)
            else:
                text, ok = QInputDialog.getText(self, 'Offset X absolute', 'Enter absolute offset value:')
                if ok:
                    xabs = True
                    xoff = float(text)

        else:
            _logger.debug("unhandled event %s", str(ddict['event']))

        if ddict['event'] in ['removeCurve', 'renameCurve', 'offsetX', 'clearChanges']:
            cdict = {
                'event': ddict['event'],
                'legend': ddict['legend']
            }
            if ddict['event'] == 'removeCurve':
                cdict['modelIndex'] = self._legendWidget.currentIndex()
            if ddict['event'] == "renameCurve":
                cdict['newLegend'] = newLegend
            if ddict['event'] == "offsetX":
                cdict['offset'] = xoff
                cdict['absolute'] = xabs
            self.sigCommonEvents.emit(cdict)

    def renameImage(self, oldLegend, newLegend):
        """
        Rename image legend.

        :param oldLegend: Old legend
        :type oldLegend: str
        :param newLegend: New legend
        :type newLegend: str
        """
        legends = self.plot.getAllImages(just_legend=True)
        for legend in legends:
            img = self.plot.getImage(legend)
            rm_legend = add_legend = legend
            if legend == oldLegend:
                rm_legend = oldLegend
                add_legend = newLegend
            self.plot.remove(rm_legend, kind='image')
            self.plot.addImage(img.getData(copy=False),
                               legend=add_legend,
                               info=img.getInfo(),
                               replace=False,
                               z=img.getZValue(),
                               selectable=img.isSelectable(),
                               draggable=img.isDraggable(),
                               colormap=img.getColormap(),
                               pixmap=None,
                               xlabel=img.getXLabel(),
                               ylabel=img.getYLabel(),
                               origin=img.getOrigin(),
                               scale=img.getScale(),
                               resetzoom=False)

    def update_alpha_all(self, alpha=1):
        """
        Update alpha value for all images

        :param alpha: Alpha value
        :type alpha: float
        """
        imgs = self.plot.getAllImages()
        for img in imgs:
            data = img.getData(copy=False)
            alpha_arr = alpha * np.ones_like(data)
            img.setData(data, alpha=alpha_arr, copy=False)

    def update_alpha(self, legend=''):
        """
        Update alpha of a curve.

        :param legend: Legend of the curve
        :type legend: str
        """
        img = self.plot.getImage(legend)
        if img is not None:
            self.update_alpha_all(alpha=self.alpha)
            data = img.getData(copy=False)
            alpha_arr = np.ones_like(data)
            img.setData(img.getData(copy=False), alpha=alpha_arr, copy=False)
        else:
            self.update_alpha_all(alpha=1)

    def offset_x(self, legend, offset, absolute=False):
        """
        Offset image along x axis.

        :param legend: Legend of the curve
        :type legend: str
        :param offset: Offset value
        :type offset: float
        :param absolute: Absolute or relative offset
        :type absolute: bool
        """
        img = self.plot.getImage(legend)
        if img is not None:
            x0, y0 = img.getOrigin()
            x1 = offset if absolute else x0 + offset
            img.setOrigin((x1, y0))
            # self.plot.resetZoom()

    def updateLegends(self, *args):
        """Sync the LegendSelector widget displayed info with the plot.
        """
        legendList = []
        for img in self.plot.getAllImages():
            legend = img.getName()
            # isActive = legend == self.plot.getActiveImage(just_legend=True)
            color = 0., 0., 0., 0.
            curveInfo = {'color': QColor.fromRgbF(*color), 'linewidth': 1, 'selected': True}
            legendList.append((legend, curveInfo))

        self._legendWidget.setLegendList(legendList)


class CurveLegendListContextMenu(LegendListContextMenu):
    def __init__(self, model):
        """
        Right click menu for curve legends

        :param model: Legend model
        :type model: silx.gui.plot.LegendSelector.LegendModel
        """
        super(CurveLegendListContextMenu, self).__init__(model=model)
        self.addAction('Change Line Width', self.changeLineWidth)
        self.addAction('Change Color', self.changeColor)
        self.addAction('Offset X', self.offsetX)
        self.addAction('Offset Data', self.offsetY)
        self.addAction('Export Displayed Data', self.exportData)
        self.addAction('Clear Changes', self.clearChanges)

    def offsetX(self):
        """
        Emit offset x position event
        """
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
            'event': "offsetX",
        }
        self.sigContextMenu.emit(ddict)

    def offsetY(self):
        """
        Emit offset x position event
        """
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
            'event': "offsetY",
        }
        self.sigContextMenu.emit(ddict)

    def exportData(self):
        """
        Emit export data event
        """
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'event': "exportData",
        }
        self.sigContextMenu.emit(ddict)

    def clearChanges(self):
        """
        Emit clear changes event
        """
        try:
            modelIndex = self.currentIdx()
            legend = str(modelIndex.data(Qt.DisplayRole))
            ddict = {
                'legend': legend,
                'event': "clearChanges",
            }
            self.sigContextMenu.emit(ddict)
        except Exception as e:
            print(e)

    def changeLineWidth(self):
        """
        Emit change line width event
        """
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
            'event': "changeLineWidth",
        }
        self.sigContextMenu.emit(ddict)

    def changeColor(self):
        """
        Emit change color event
        """
        modelIndex = self.currentIdx()
        legend = str(modelIndex.data(Qt.DisplayRole))
        ddict = {
            'legend': legend,
            'label': legend,
            'selected': modelIndex.data(qt.Qt.CheckStateRole),
            'type': str(modelIndex.data()),
            'event': "changeColor",
        }
        self.sigContextMenu.emit(ddict)


class ImageLegendListContextMenu(LegendListContextMenu, QMenu):
    def __init__(self, model):
        """
        Right click menu for image legends

        :param model: Legend model
        :type model: silx.gui.plot.LegendSelector.LegendModel
        """
        super(ImageLegendListContextMenu, self).__init__(model=model) # Do not init with LegendListContextMenu
        # QMenu.__init__(self)
        # self.model = model
        #
        # self.addAction('Set Active', self.setActiveAction)
        # self.addAction('Remove curve', self.removeItemAction)
        # self.addAction('Rename curve', self.renameItemAction)
        #
        # self.addAction('Offset X', self.offsetX)
        # self.addAction('Clear changes', self.clearChanges)

    # def exec_(self, pos, idx):
    #     """
    #     Store current index.
    #     :param pos: Current legend drawn pos
    #     :type pos: float, float
    #     :param idx: Current index
    #     :type idx: int
    #     """
    #     self.__currentIdx = idx
    #     super(ImageLegendListContextMenu, self).popup(pos)

    # def currentIdx(self):
    #     """
    #     Get current selected legend index
    #     :return: Current index
    #     :rtype: int
    #     """
    #     return self.__currentIdx

    def offsetX(self):
        """
        Emit offset x position event
        """
        try:
            modelIndex = self.currentIdx()
            legend = str(modelIndex.data(Qt.DisplayRole))
            ddict = {
                'legend': legend,
                'label': legend,
                'selected': modelIndex.data(qt.Qt.CheckStateRole),
                'type': str(modelIndex.data()),
                'event': "offsetX",
            }
            self.sigContextMenu.emit(ddict)
        except Exception as e:
            print(e)

    def clearChanges(self):
        """
        Emit clear changes event
        """
        try:
            modelIndex = self.currentIdx()
            legend = str(modelIndex.data(Qt.DisplayRole))
            ddict = {
                'legend': legend,
                'event': "clearChanges",
            }
            self.sigContextMenu.emit(ddict)
        except Exception as e:
            print(e)
