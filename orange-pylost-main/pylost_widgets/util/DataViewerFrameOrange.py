# coding=utf-8
"""
Customized data viewer based on 'silx' package, to display raw data/ plot data
"""
import copy
import os
from time import sleep

import numpy as np
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QHBoxLayout, \
    QInputDialog, QLabel, QLineEdit, QPushButton, QSizePolicy, QStyle, QTableWidgetItem, QTextEdit, QWidget
from astropy.units import Quantity
from silx.gui.data import DataViews
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector
from silx.gui.plot.CurvesROIWidget import ROI
from silx.gui.plot.MaskToolsWidget import MaskToolsWidget
from silx.utils.weakref import WeakMethodProxy

from PyLOSt.algorithms.util.util_math import pv, rms
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.resource_path import resource_path
from pylost_widgets.util.util_functions import alertMsg, plot_win_colormap
from pylost_widgets.util.util_plots import OrangePlot2D

qtCreatorFile = resource_path(os.path.join("gui", "mask_widgets.ui"))  # Enter file here.


# UI_mask_cls, UI_mask_widget = uic.loadUiType(qtCreatorFile)

class DataViewerFrameOrange(DataViewerFrame):
    infoChanged = pyqtSignal()
    sigMaskParams = pyqtSignal(float, float, float, float, bool, bool)

    def __init__(self, parent=None, editInfo=False, show_mask=True):
        """
        Constructor for DataViewerFrameOrange class. It initializes mask and roi tools, info view, preview etc.

        :param parent: Parent widget
        :type parent: QWidget
        :param editInfo: Allow editing of information such as pixel size (for type MetrologyData)
        :type editInfo: bool
        :param show_mask: Show/hide mask icon
        :type show_mask: bool

        """
        super(DataViewerFrameOrange, self).__init__(parent)
        self.cmap_name = 'turbo'
        self.plot_win = None
        self.has_updated_plot = False
        self.has_added_factor = False
        self.factor = 0
        self.mask_widgets = None
        self.mask_tools = self.get_mask_tools()
        self.roi_tools = self.get_roi_tools()
        self.center_data = (0, 0)
        self.center_absolute = (0, 0)
        self.move_rect = False
        self.show_mask = show_mask

        self.displayedViewChanged.connect(self.changeDisplayView)
        self.dataChanged.connect(self.changeData)
        stack = self.findChild(qt.QStackedWidget)
        self.infoview = _InfoView(parent=stack, dataViewer=self, editable=editInfo, infoChanged=self.infoChanged)
        self.preview = _Preview(parent=stack, dataViewer=self)

        self.addView(self.preview)
        self.addView(self.infoview)

        self.__numpySelection = self.children()[0].findChild(NumpyAxesSelector)
        self.__numpySelection.selectedAxisChanged.connect(self.__numpyAxisChanged)
        self.__numpySelection.selectionChanged.connect(self.__numpySelectionChanged)
        self.__numpySelection.customAxisChanged.connect(self.__numpyCustomAxisChanged)

        try:
            parent.sigMaskParams.connect(self.update_mask_params_sig)
        except Exception:
            pass

    def get_roi_tools(self):
        """
        Get roi tools used in masking 1D data.

        :return: Roi tools widget
        :rtype: silx.gui.plot.CurvesROIWidget.CurvesROIWidget
        """
        try:
            view = self.getViewFromModeId(DataViews.PLOT1D_MODE)
            if view is not None:
                return view.getWidget().getCurvesRoiWidget()
        except Exception as e:
            pass
        return None

    def get_mask_tools(self):
        """
        Get mask tools widget used for masking 2D data.

        :return: Mask tools widget
        :rtype: silx.gui.plot.MaskToolsWidget.MaskToolsWidget
        """
        try:
            view = self.getViewFromModeId(DataViews.IMAGE_MODE)
            if view is not None:
                return view.getWidget().findChild(MaskToolsWidget)
        except Exception as e:
            pass
        return None

    def update_mask_params_sig(self, cx, cy, w, h, view_pixels, relative):
        """
        Update mask parameters center, size, flags (view pixels, use relative center).

        :param cx: Center x
        :type cx: float
        :param cy: Center y
        :type cy: float
        :param w: Width
        :type w: float
        :param h: Height
        :type h: float
        :param view_pixels: Flag to use params in pixel values
        :type view_pixels: bool
        :param relative: Flag to use center relative to current data size
        :type relative: bool
        """
        self.blockSignals(True)
        if self.mask_widgets is not None:
            self.mask_widgets.cbUnits.blockSignals(True)
            self.mask_widgets.cbRelative.blockSignals(False)
            self.mask_widgets.cbUnits.setChecked(view_pixels)
            self.mask_widgets.cbRelative.setChecked(relative)
            self.set_mask_inputs(cx, cy, w, h)
            self.update_center_info()
            self.mask_widgets.cbUnits.blockSignals(False)
            self.mask_widgets.cbRelative.blockSignals(False)
        self.blockSignals(False)

    def __numpyAxisChanged(self):
        pass

    def __numpyCustomAxisChanged(self, name, value):
        pass

    def __numpySelectionChanged(self):
        """
        The function is called when data slice item is selected along any axis (e.g. with the sliders).
        """
        try:
            if isinstance(self.displayedView(), DataViews._ImageView):
                self._update_posinfo()
                self.updateColormaps()
                self._update_xy_image()
            elif isinstance(self.displayedView(), DataViews._Plot1dView):
                self._update_xy_curve()
            QtWidgets.qApp.processEvents()
        except Exception as e:
            print(e)

    def setCmapName(self, name):
        """
        Set colormap name for 2d plots

        :param name: Colormap name
        :type name: str
        """
        self.cmap_name = name

    def changeData(self):
        """
        Callback when data in viewer changes
        """
        self.changeDisplayView()

    def changeDisplayView(self):
        """
        The dataviewer shows multiple views (raw / curve / image etc.).
        This function is called whenever a different view other than the displayed, is clicked.
        """
        try:
            if isinstance(self.displayedView(), DataViews._ImageView):
                iv = self.displayedView()
                self.plot_win = iv.getWidget().children()[1]  # iv.getWidget().findChild(PlotWindow)
                QtWidgets.qApp.processEvents()
                self._update_gui_image()

                if self.plot_win is not None:
                    self.plot_win.getMaskAction().setVisible(self.show_mask)
                    positionInfo = self.plot_win.getPositionInfoWidget()
                    if isinstance(self.data(), MetrologyData) and self.data().has_flag(
                            'invert_y_axis') and self.data().get_flag('invert_y_axis'):
                        self.plot_win.setYAxisInverted(True)  # set origin to top left
                    else:
                        self.plot_win.setYAxisInverted(False)

                    if not self.has_updated_plot:
                        # Update colormap action dialog with factor parameter
                        self.plot_win.colormapAction.triggered.connect(self.colormapDialogOpened)
                        if positionInfo is not None:
                            self.addPositionInfo(positionInfo)
                        self.loadMaskTools()
                        self.has_updated_plot = True
                    if positionInfo is not None:
                        positionInfo.updateInfo()
                    self.plot_win.resetZoom()
            elif isinstance(self.displayedView(), DataViews._Plot1dView):
                iv = self.displayedView()
                self.plot_1d = iv.getWidget()  # Pot 1D
                QtWidgets.qApp.processEvents()
                self.__numpySelection.setAxisNames(['x'])
                self._update_xy_curve()
        except Exception as e:
            print(e)

    def _update_posinfo(self):
        """
        Update current XY position and data value information in th 2d image, coresponding to the mouse hover position.
        The displayed values also include data rms, peak to valley, standard deviation and data dimensions which don't change with the mouse hover.
        """
        if self.plot_win is not None:
            positionInfo = self.plot_win.getPositionInfoWidget()
            if positionInfo is not None:
                positionInfo.updateInfo()

    def addPositionInfo(self, positionInfo):
        """
        Add additional parameters to display in the traditional live X/Y/Data display widget, such as data rms, pv, std and dimensions.

        :param positionInfo: Widget containing position information
        :type positionInfo: silx.gui.plot.tools.PositionInfo
        """
        pwLayout = self.plot_win.centralWidget().layout()
        if type(pwLayout) is QGridLayout:
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
        else:
            layout = positionInfo.layout()
        converters = [
            ('Rms', WeakMethodProxy(self._getImageRms)),
            ('PV', WeakMethodProxy(self._getImagePV)),
            ('Std', WeakMethodProxy(self._getImageStd)),
        ]
        if isinstance(self.data(), MetrologyData):
            converters += [('RealDims', WeakMethodProxy(self._getImageDims)), ]
        # Create all QLabel and store them with the corresponding converter
        for name, func in converters:
            layout.addWidget(QLabel('<b>' + name + ':</b>'))
            contentWidget = QLabel()
            contentWidget.setText('------')
            contentWidget.setTextInteractionFlags(Qt.TextSelectableByMouse)
            contentWidget.setFixedWidth(
                contentWidget.fontMetrics().boundingRect('############').width())
            layout.addWidget(contentWidget)
            positionInfo._fields.append((contentWidget, name, func))

        if type(pwLayout) is QGridLayout:
            layout.addStretch(1)
            bottomStats = QWidget(None)
            bottomStats.setLayout(layout)
            pwLayout.addWidget(bottomStats, 2, 0, 1, -1)

    def _update_xy_curve(self):
        """
        Update start position, x-scaling and xy labels of all curves in the curve view.
        """
        try:
            if self.plot_1d is not None:
                if isinstance(self.data(), MetrologyData):
                    QtWidgets.qApp.processEvents()
                    sleep(0.1)
                    curves = self.plot_1d.getAllCurves()  # getActiveCurve()
                    for curve in curves:
                        if curve is not None:
                            self._set_curve_scale(curve)
                            QtWidgets.qApp.processEvents()
                    if any(curves):
                        self._set_curve_label()
                        self.plot_1d.resetZoom()
                    QtWidgets.qApp.processEvents()
        except Exception as e:
            print(e)

    def _set_curve_scale(self, curve):
        """
        Update start position and x-scaling (e.g. pix to mm) of selected curve

        :param curve: Selected curve object
        :type curve: silx.gui.plot.items.curve.Curve
        """
        pix_sz_sel, units_sel, axis_sel, axis_val_sel, start_sel = self.get_sel_data_attr()
        x, y, xerr, yerr = curve.getData()
        if isinstance(axis_val_sel[-1], Quantity) and np.any(axis_val_sel[-1].value) and len(axis_val_sel[-1]) == len(
                y):
            x = axis_val_sel[-1].value
        else:
            x = x * pix_sz_sel[-1].value
        x = x + start_sel[-1].value
        curve.setData(x, y, xerr, yerr)

    def _set_curve_label(self):
        """
        Update the xy labels including their units
        """
        pix_sz_sel, units_sel, axis_sel, axis_val_sel, start_sel = self.get_sel_data_attr()
        if isinstance(axis_val_sel[-1], Quantity) and np.any(axis_val_sel[-1].value):
            xlabel = axis_sel[-1] + ' ({})'.format(axis_val_sel[-1].unit)
        else:
            xlabel = axis_sel[-1] + ' ({})'.format(units_sel[-1])
        ylabel = 'Data ({})'.format(self.data().unit)
        self.plot_1d.setGraphXLabel(xlabel)
        self.plot_1d.setGraphYLabel(ylabel)

    def _update_gui_image(self):
        """
        Update image params (scale, labels etc) and colormaps
        """
        self._update_xy_image()
        self.updateColormaps()

    def _update_xy_image(self):
        """
        Update Image start position, scale and xy labels
        """
        try:
            if self.plot_win is not None:
                if isinstance(self.data(), MetrologyData):
                    QtWidgets.qApp.processEvents()
                    image = self.plot_win.getActiveImage()
                    if image is not None:
                        self._set_xy_scale_label(image)
                        self.plot_win.resetZoom()
        except Exception as e:
            print(e)

    def _set_xy_scale_label(self, image):
        """
        Update Image start position, scale and xy labels in the 2d plot

        :param image: Image object
        :type image: silx.gui.plot.items.image.ImageData
        """
        pix_sz_sel, units_sel, axis_sel, axis_val_sel, start_sel = self.get_sel_data_attr()

        start = [x.value for x in start_sel[::-1]]  # [0,0]
        # o = self.data().start_position
        # if o is not None and any(o):
        #     start = [x.value for x in o[::-1]]

        xy = [pix_sz_sel[-1].value, pix_sz_sel[-2].value]  # xy=(x,y)
        if len(axis_val_sel[-1]) == image.getData().shape[-1]:
            xy[0] = np.nanmean(np.diff(axis_val_sel[-1].value))
            # start[0] = axis_val_sel[-1].value[0]
        if len(axis_val_sel[-2]) == image.getData().shape[-2]:
            xy[1] = np.nanmean(np.diff(axis_val_sel[-2].value))
            # start[1] = axis_val_sel[-2].value[0]
        if isinstance(axis_val_sel[-1], Quantity) and np.any(axis_val_sel[-1].value):
            xlabel = axis_sel[-1] + ' ({})'.format(axis_val_sel[-1].unit)
            ylabel = axis_sel[-2] + ' ({})'.format(axis_val_sel[-2].unit)
        else:
            xlabel = axis_sel[-1] + ' ({})'.format(units_sel[-1])
            ylabel = axis_sel[-2] + ' ({})'.format(units_sel[-2])

        image.setScale(tuple(xy))
        image.setOrigin(tuple(start))
        self.plot_win.setGraphXLabel(xlabel)
        self.plot_win.setGraphYLabel(ylabel)

    def get_sel_data_attr(self):
        """
        Get selection parameters like pixel size, units, axis names, axis values, start position.

        :return: pixel size, units, axes, axes values, start positions
        :rtype: list[Quantity[float]], list[str], np.ndarray[str], list[Quantity[np.ndarray]], list[Quantity[float]]
        """
        pix_sz = self.data().pix_size
        pix_sz_sel = self.get_selection(pix_sz)
        units = [str(x.unit) for x in pix_sz]
        units_sel = self.get_selection(units)
        axis_names = np.asarray(self.data().axis_names)
        axis_sel = self.get_selection(axis_names)
        axis_val_sel = self.get_selection(self.data().get_axis_val_items())
        start = [self.data().index_list[i][0] * pix_sz[i] for i in np.arange(self.data().ndim)]
        start_sel = self.get_selection(start)

        return pix_sz_sel, units_sel, axis_sel, axis_val_sel, start_sel

    def get_selection_axes(self):
        """
        Get selected and permuted axes.

        :return: selection axes, permutations axes
        :rtype: list[int], list[int]
        """
        selection = list(self.__numpySelection.selection())
        permutation = list(self.__numpySelection.permutation())
        return selection, permutation

    def get_selection(self, arr):
        """
        Apply selection and permutation of axes to an array data.

        :param arr: Input array data (e.g. axis names)
        :type arr: np.ndarray or list
        :return: Output array data
        :rtype: np.ndarray or list
        """
        selection = list(self.__numpySelection.selection())
        selection_idx = [i for i, sel in enumerate(selection) if type(sel) is slice]
        permutation = list(self.__numpySelection.permutation())
        if type(arr) is np.ndarray:
            return arr[selection_idx][permutation]
        else:
            arr_sel = [arr[i] for i in selection_idx]
            return [arr_sel[i] for i in permutation]

    def _getImageRms(self, *args):
        """Get image rms stats"""
        return self._getImageStats(stype='rms')

    def _getImagePV(self, *args):
        """Get image peak to valley stats"""
        return self._getImageStats(stype='pv')

    def _getImageStd(self, *args):
        """Get image std stats"""
        return self._getImageStats(stype='std')

    def _getImageDims(self, *args):
        """Get image dimensions"""
        return self._getImageStats(stype='dims')

    def _getImageStats(self, stype='rms'):
        """
        Get statistics of data like rms, peak to valley, standard deviation, dimensions

        :param stype: Type of statistics
        :type stype: str
        :return: Formatted stats string
        :rtype: str
        """
        try:
            if self.plot_win is not None:
                activeImage = self.plot_win.getActiveImage()
                if (activeImage is not None and
                        activeImage.getData(copy=False) is not None):
                    data = activeImage.getData(copy=False)
                    if stype == 'rms':
                        val = rms(data)
                    elif stype == 'pv':
                        val = pv(data)
                    elif stype == 'std':
                        val = np.nanstd(data)
                    elif stype == 'dims' and isinstance(self.data(), MetrologyData):
                        val = 'x'.join(['{:.1f}'.format(x) for x in self.data().size_detector[::-1]])
                    else:
                        val = '-'
                    if isinstance(self.data(), MetrologyData) and stype in ['rms', 'pv', 'std']:
                        val = '{:.4f} {}'.format(val, self.data().unit)
                    return val
            else:
                return '-'
        except Exception as e:
            print(e)
            return '-'

    def updateColormaps(self):
        """
        Update colormap of the active image in the plot window.
        """
        try:
            if self.plot_win is not None:
                QtWidgets.qApp.processEvents()
                sleep(0.1)
                image = self.plot_win.getActiveImage()
                if image is not None:
                    QtWidgets.qApp.processEvents()
                    self.setImageColormap(image)
        except Exception as e:
            print(e)

    def setImageColormap(self, image, base='ra'):
        """
        Set image colormap. If the colormap is based on turbo, the colormap values are recalculated based on image data
        (e.g. rescaling within k-std or mad, for better visualization of surface features)

        :param image: Active image object
        :type image: silx.gui.plot.items.image.ImageData
        :param base: Type of rescaling for turbo colormaps
        :type base: str
        """
        if self.factor is not None:
            if self.factor < 0.001:
                self.factor = None
        lsc = plot_win_colormap(image.getData(copy=False), cmap_name=self.cmap_name, base=base, factor=self.factor)
        image.setColormap(lsc)

    def colormapDialogOpened(self):
        """
        Edit native silx colormap dialog, to include a edit box for 'factor' which is used for rescaling colormap data (e.g. rescale values < factor * std from median)
        """
        try:
            if self.plot_win is not None and not self.has_added_factor:
                self.dialogColormap = self.plot_win.colormapAction._dialog
                self.leFactor = QtWidgets.QLineEdit('2.0')
                self.leFactor.setValidator(QDoubleValidator())
                self.leFactor.textChanged.connect(self.changeFactor)
                self.dialogColormap.layout().insertRow(3, 'Factor:', self.leFactor)
                self.has_added_factor = True
        except Exception as e:
            print(e)

    def changeFactor(self):
        """
        Callback when 'factor' is changed. Color map is updated with the new factor each time.
        """
        try:
            if self.leFactor.text() != '':
                self.factor = float(self.leFactor.text())
                if 'urbo_' in self.dialogColormap._comboBoxColormap.currentText():
                    self.updateColormaps()
        except Exception as e:
            print(e)

    def checkMaskUnits(self):
        """
        Callback of checking/unchecking 'view pixels' in mask dialog in 2d image view. If checked the mask position / size is shown in pixels, and if unchecked they are shown in detector pixel units
        """
        try:
            if isinstance(self.data(), MetrologyData):
                self.update_mask_units()
                self.update_center_info()
        except Exception as e:
            print(e)

    def update_mask_units(self):
        """Update mask center/size values to pixels or pix units, given by corresponding checkbox"""
        sx, sy = self.mask_tools._scale
        cx, cy, w, h = self.get_mask_inputs()
        if self.mask_widgets.cbUnits.isChecked():
            self.set_mask_inputs(int(cx / sx), int(cy / sy), int(w / sx), int(h / sy))
        else:
            self.set_mask_inputs(cx * sx, cy * sy, w * sx, h * sy)

    def update_center_info(self):
        """Update mask center information. Center values are updated relative to current data size or total cam size, given by corresponding checkbox"""
        if not isinstance(self.data(), MetrologyData):
            return
        if self.mask_widgets is not None:
            axes = self.get_selection(self.data().axis_names)
            pix_size = self.get_selection(self.data().pix_size)
            if self.mask_widgets.cbUnits.isChecked():
                self.center_data = self.data().center_pix[::-1]
                self.center_absolute = self.data().center_absolute_pix[
                                       ::-1] if self.data().center_absolute_pix is not None else [0, 0]
                self.mask_widgets.label_x.setText('{} (pix)'.format(axes[-1]))
                self.mask_widgets.label_y.setText('{} (pix)'.format(axes[-2]))
            else:
                self.center_data = [x.value for x in self.data().center][::-1]
                self.center_absolute = [x.value for x in self.data().center_absolute][
                                       ::-1] if self.data().center_absolute is not None else [0, 0]
                self.mask_widgets.label_x.setText('{} ({})'.format(axes[-1], pix_size[-1].unit))
                self.mask_widgets.label_y.setText('{} ({})'.format(axes[-2], pix_size[-2].unit))
            self.set_center_info()

    def set_center_info(self):
        """
        Display center of image information, for the current image size if checkbox Relative to data center' is checked, else center is shown for the initial image size
        """
        pix_size = [Quantity(1, 'pixel'), Quantity(1, 'pixel')]
        try:
            pix_size = self.get_selection(self.data().pix_size)
        except Exception as e:
            print(e)
        if self.mask_widgets.cbUnits.isChecked():
            unitStr = ''
            fmt = '{}'
        else:
            unitStr = '{} '.format(pix_size[-1].unit)
            fmt = '{:.3f}'
        c = self.center_data if self.mask_widgets.cbRelative.isChecked() else self.center_absolute
        ctyp = 'data' if self.mask_widgets.cbRelative.isChecked() else 'cam'
        txt = 'Center {} {}({})'.format(' x '.join([fmt.format(x) for x in c]), unitStr, ctyp)
        self.mask_widgets.info_center.setText(txt)
        self.mask_widgets.info_center.setToolTip(txt)

    def get_mask_inputs(self):
        """
        Get mask center position (x, y) and size (width, height). These data are retrieved from mask dialog edit boxes in the 2d image view.

        :return: center x, center y, width, height
        :rtype: float, float, float, float
        """
        cx = cy = w = h = 0
        try:
            cx = float(self.mask_widgets.leCX.text())
            cy = float(self.mask_widgets.leCY.text())
            w = float(self.mask_widgets.leSX.text())
            h = float(self.mask_widgets.leSY.text())
        except Exception as e:
            pass
        return cx, cy, w, h

    def set_mask_inputs(self, cx, cy, w, h):
        """
        Set mask center position (x, y) and size (width, height)

        :param cx: Center x
        :type cx: float
        :param cy: Center y
        :type cy: float
        :param w: Width
        :type w: float
        :param h: Height
        :type h: float
        """
        fmt = '{:.3f}'
        if self.mask_widgets.cbUnits.isChecked():
            cx, cy, w, h = (int(np.round(cx)), int(np.round(cy)), int(np.ceil(w)), int(np.ceil(h)))
            fmt = '{}'
        self.mask_widgets.leCX.setText(fmt.format(cx))
        self.mask_widgets.leCY.setText(fmt.format(cy))
        self.mask_widgets.leSX.setText(fmt.format(w))
        self.mask_widgets.leSY.setText(fmt.format(h))

    def change_mask_params(self):
        """Emit mask inputs signal"""
        cx, cy, w, h = self.get_mask_inputs()
        self.sigMaskParams.emit(cx, cy, w, h, self.mask_widgets.cbUnits.isChecked(),
                                self.mask_widgets.cbRelative.isChecked())

    def connect_mask_params(self):
        """Callback change_mask_params when any mask input is changed"""
        self.mask_widgets.leCX.textChanged.connect(self.change_mask_params)
        self.mask_widgets.leCY.textChanged.connect(self.change_mask_params)
        self.mask_widgets.leSX.textChanged.connect(self.change_mask_params)
        self.mask_widgets.leSY.textChanged.connect(self.change_mask_params)
        self.mask_widgets.cbUnits.toggled.connect(self.change_mask_params)
        self.mask_widgets.cbRelative.toggled.connect(self.change_mask_params)

    # Mask tools
    def loadMaskTools(self):
        """
        Additional tools for drawing masks in 2d image view, such as drawing with given center and size.
        """
        try:
            if self.plot_win is not None:
                mtd = self.plot_win.getMaskToolsDockWidget()
                self.mask_tools = mtd.findChild(MaskToolsWidget)
                if self.mask_widgets is None:
                    self.mask_widgets = uic.loadUi(qtCreatorFile)
                    self.mask_tools.layout().addWidget(self.mask_widgets)
                    self.mask_tools.layout().setAlignment(self.mask_widgets, Qt.AlignLeft)
                    self.mask_tools.drawActionGroup.triggered.connect(self.disableDraw)
                    self.mask_tools.rectAction.triggered.connect(self.enableDraw)
                    self.mask_widgets.hide()
                    self.mask_widgets.btnDraw.clicked.connect(self.drawRectMask)
                    self.mask_widgets.btnMove.clicked.connect(self.moveRectMask)
                    self.mask_tools.plot.sigPlotSignal.connect(self._maskPlotDrawEvent)
                    self.mask_widgets.cbUnits.toggled.connect(self.checkMaskUnits)
                    self.mask_widgets.cbRelative.toggled.connect(self.set_center_info)
                    self.checkMaskUnits()
                    if isinstance(self.data(), MetrologyData):
                        self.mask_widgets.cbUnits.show()
                    else:
                        self.mask_widgets.cbUnits.hide()
                    self.connect_mask_params()
        except Exception as e:
            print(e)

    def showMaskTools(self):
        """
        Show / hide mask tools widget
        """
        if self.plot_win is not None:
            mtd = self.plot_win.getMaskToolsDockWidget()
            mtd.setVisible(True)

    def update_mask_params(self, cx, cy, w, h):
        """
        Update mask parameters (size and center).

        :param cx: Center x
        :type cx: float
        :param cy: Center y
        :type cy: float
        :param w: Width
        :type w: float
        :param h: Height
        :type h: float
        """
        self.mask_widgets.cbUnits.setChecked(False)
        self.mask_widgets.cbRelative.setChecked(True)
        self.set_mask_inputs(cx, cy, w, h)

    def _maskPlotDrawEvent(self, event):
        """
        Callback after graphically drawing mask over image canvas.

        :param event: Draw event
        :type event: QEvent
        """
        if (self.mask_tools._drawingMode is None or
                event['event'] not in ('drawingProgress', 'drawingFinished')):
            return
        if (self.mask_tools._drawingMode == 'rectangle' and
                event['event'] == 'drawingFinished'):

            ox, oy = self.mask_tools._origin
            sx, sy = self.mask_tools._scale if self.mask_widgets.cbUnits.isChecked() else [1, 1]

            height = abs(event['height'] / sy)
            width = abs(event['width'] / sx)
            cy = (event['y'] / sy) + 0.5 * height
            cx = (event['x'] / sx) + 0.5 * width
            if sy < 0:
                cy -= height
            if sx < 0:
                cx -= width

            if self.mask_widgets.cbRelative.isChecked():
                cy = cy - self.center_data[1]
                cx = cx - self.center_data[0]
            else:
                cy = cy - self.center_absolute[1]
                cx = cx - self.center_absolute[0]

            self.set_mask_inputs(cx, cy, width, height)

    def enableDraw(self):
        """Show mask tools widget"""
        self.mask_widgets.show()

    def disableDraw(self):
        """Hide mask tools widget"""
        self.mask_widgets.hide()

    def drawROI(self, cx=0, w=0, check_relative=False):
        """
        Draw region of interest for 1D curve

        :param cx: Center x
        :type cx: float
        :param w: Width
        :type w: float
        :param check_relative: Relative to current data size
        :type check_relative: bool
        """
        try:
            if self.roi_tools is not None and w > 0:
                # if check_relative:
                #     cx = cx + self.data().center[-1]
                # else:
                #     cx = cx + self.data().center_absolute[-1]
                view = self.getViewFromModeId(DataViews.PLOT1D_MODE)
                if view is not None:
                    plt = view.getWidget()
                    xmin, xmax = plt.getXAxis().getLimits()
                    cx = cx + (xmax + xmin) / 2

                start = cx - (w / 2)
                end = cx + (w / 2)
                roi = ROI('Mask')
                roi.setType('X')
                roi.setFrom(start)
                roi.setTo(end)
                self.roi_tools.roiTable.clear()
                self.roi_tools.roiTable.addRoi(roi)
        except Exception as e:
            print('drawROI <- Mask')
            print(e)

    def drawRectMask(self, cx=None, cy=None, w=None, h=None, check_pixels=None, check_relative=None):
        """
        Draw mask using the provided center and size values.

        :param cx: Center x
        :type cx: float
        :param cy: Center y
        :type cy: float
        :param w: Width
        :type w: float
        :param h: Height
        :type h: float
        :param check_pixels: Use values in pixels
        :type check_pixels: bool
        :param check_relative: Relative to current data size
        :type check_relative: bool
        """
        try:
            if self.mask_widgets.leCX.text() == '' or self.mask_widgets.leCY.text() == '':
                alertMsg('Invalid input', 'Please enter center position')
                return
            if self.mask_widgets.leSX.text() == '' or self.mask_widgets.leSY.text() == '':
                alertMsg('Invalid input', 'Please enter size')
                return

            c = (float(self.mask_widgets.leCX.text()), float(self.mask_widgets.leCY.text())) if cx is None or cy is None else (cx, cy)
            sz = (float(self.mask_widgets.leSX.text()), float(self.mask_widgets.leSY.text())) if w is None or h is None else (w, h)
            _pixels = self.mask_widgets.cbUnits.isChecked() if check_pixels is None else check_pixels
            _relative = self.mask_widgets.cbRelative.isChecked() if check_relative is None else check_relative

            if _relative:
                c = (c[0] + self.center_data[0], c[1] + self.center_data[1])
            else:
                c = (c[0] + self.center_absolute[0], c[1] + self.center_absolute[1])
            self.draw_rect(c, sz, view_pixels=_pixels)
        except Exception as e:
            print('drawRectMask <- Mask')
            print(e)

    def draw_rect(self, c, sz, view_pixels=False):
        """
        Draw mask using the provided center and size values.

        :param c: Center (x,y)
        :type c: tuple[float, float]
        :param sz: Size (w,h)
        :type sz: tuple[float, float]
        :param view_pixels: Use values in pixels
        :type view_pixels: bool
        """
        try:
            mask = self.mask_tools.getSelectionMask()
            s = (c[0] - (sz[0] / 2), c[1] - (sz[1] / 2))
            s = np.asarray(s)
            sz = np.asarray(sz)
            o = np.asarray(self.mask_tools._origin)
            scale = np.asarray(self.mask_tools._scale)
            if view_pixels or not isinstance(self.data(), MetrologyData):  # already in pixel
                shape = self.data().shape[::-1]
                s = s - (o / scale)
                if sz[0] % 2 > 0 or shape[0] % 2 > 0:  # odd
                    s[0] = s[0] + 1
                if sz[1] % 2 > 0 or shape[1] % 2 > 0:  # odd
                    s[1] = s[1] + 1
            else:
                orig = (s - o) - scale / 24
                size = sz - scale / 24
                axes = self.data().get_axis_val_items()
                if len(axes) <= 1:
                    x = self.data().get_axis_val_items()[-1]
                else:
                    y, x = self.data().get_axis_val_items()[-2:]
                s = x.value.searchsorted(orig[0]), y.value.searchsorted(orig[1])
                sz = x.value.searchsorted(size[0]), y.value.searchsorted(size[1])
            # s = np.floor(s).astype(int)
            # sz = np.floor(sz).astype(int)
            if sz[0] > 0 and sz[1] > 0:
                if mask is None:
                    shape = self.data().shape
                    if len(shape) > 2:
                        shape = shape[-2:]
                    mask = np.zeros(shape).view(np.ndarray)  # feed silx with 2D mask only
                mask[:, :] = 0
                mask[slice(int(max(0, s[1])), int(s[1] + sz[1])), slice(int(max(0, s[0])), int(s[0] + sz[0]))] = 1  # self.mask_tools.levelSpinBox.value()
                self.mask_tools.setSelectionMask(mask)
        except Exception as e:
            print('draw_rect <- Mask')
            print(e)

    def getMaskPosSize(self):
        """
        Get mask size and position in pixel length units.

        :return: Center x, Center y, Width, Height
        :rtype: float, float, float, float
        """
        cx, cy, w, h = self.get_mask_inputs()
        if self.mask_widgets.cbUnits.isChecked():
            sx, sy = self.mask_tools._scale
            return cx * sx, cy * sy, w * sx, h * sy
        else:
            return cx, cy, w, h

    def getMask(self):
        """
        Get mask array

        :return: Mask 2D array
        :rtype: np.ndarray
        """
        if self.mask_tools is not None:
            return self.mask_tools.getSelectionMask()
        else:
            return None

    def moveRectMask(self):
        """
        Enable/disable move mask option.
        """
        if self.move_rect:
            self.disable_move_rect()
        else:
            QApplication.setOverrideCursor(Qt.SizeAllCursor)
            self.move_rect = True

    def disable_move_rect(self):
        """
        Disable move mask option.
        """
        if self.move_rect:
            QApplication.restoreOverrideCursor()
            self.move_rect = False

    def move_center_px(self, dx, dy):
        """
        Move center by given pixels.

        :param dx: pixels along x
        :type dx: int
        :param dy: pixels along y
        :type dy: int
        """
        try:
            if dx or dy:
                cx, cy, w, h = self.get_mask_inputs()
                sx, sy = [1, 1] if self.mask_widgets.cbUnits.isChecked() else self.mask_tools._scale
                self.set_mask_inputs(cx + dx * sx, cy + dy * sy, w, h)
                if w > 0 and h > 0:
                    self.drawRectMask()
        except Exception as e:
            pass

    def mousePressEvent(self, event):
        """
        Disable move mask option (if it is on), with mouse click anywhere.

        :param event: Mouse press event object
        :type event: QEvent
        """
        self.disable_move_rect()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        """
        Move mask based on key presses for numbers 2,4,6,8.

        :param event: Key press event object
        :type event: QEvent
        """
        if self.move_rect:
            if event.key() == QtCore.Qt.Key_Q:
                self.disable_move_rect()
            elif event.key() == QtCore.Qt.Key_Left or event.key() == QtCore.Qt.Key_4:
                self.move_center_px(-1, 0)
            elif event.key() == QtCore.Qt.Key_Right or event.key() == QtCore.Qt.Key_6:
                self.move_center_px(1, 0)
            elif event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_8:
                self.move_center_px(0, 1)
            elif event.key() == QtCore.Qt.Key_Down or event.key() == QtCore.Qt.Key_2:
                self.move_center_px(0, -1)
        super().keyPressEvent(event)


from Orange.data.io import FileFormat, class_from_qualified_name
from orangewidget.utils.filedialogs import open_filename_dialog
from Orange.data import Table


class MotorDialog(QDialog):
    MOTOR_NAMES = ['', 'motor_X', 'motor_Y', 'motor_Z', 'motor_RX', 'motor_RY', 'motor_RZ']

    def __init__(self, parent=None, title="Add / edit motors", sel_motor='', data=None):
        """
        Update motor values dialog.

        :param parent: Parent object
        :type parent: QWidget
        :param title: Dialog title
        :type title: str
        :param sel_motor: Selected motor (X/Y/Z/RX/RY/RZ)
        :type sel_motor: str
        :param data: Primary slopes/shape data to which the motor values are associated
        :type data: MetrologyData
        """
        QDialog.__init__(self, parent)
        self.setWindowTitle(title)
        self.data = data
        self.motors = data.motors if data is not None else []
        self.motors_original = copy.deepcopy(data.motors)
        self.data_shape = data.shape if data is not None else ()

        self.cb_motor = QComboBox(self)
        self.btn_file = QPushButton(text='...', icon=self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_file.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))
        self.le_unit = QLineEdit(self)

        hw = QWidget(self)
        hw.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        hl = QHBoxLayout(self)
        hl.setAlignment(Qt.AlignLeft)
        hw.setLayout(hl)

        lbl = QLabel('Start : ')
        lbl.setFixedWidth(30)
        hl.addWidget(lbl)
        self.start_m = QLineEdit()
        self.start_m.setFixedWidth(100)
        self.start_m.editingFinished.connect(self.change_start)
        hl.addWidget(self.start_m)
        lbl = QLabel('Step : ')
        lbl.setFixedWidth(30)
        hl.addWidget(lbl)
        self.step_m = QLineEdit()
        self.step_m.setFixedWidth(100)
        self.step_m.editingFinished.connect(self.change_step)
        hl.addWidget(self.step_m)
        btn = QPushButton(text='Reload from data')
        btn.clearFocus()
        btn.clicked.connect(self.reload_motor)
        hl.addWidget(btn)

        self.te_vals = QTextEdit(self)
        self.info = QLabel(self)
        self.info.setStyleSheet('color:red')
        buttonBox = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close, self)

        layout = QFormLayout(self)
        layout.addRow('Motor :', self.cb_motor)
        layout.addRow('Load from file:', self.btn_file)
        layout.addRow('Motor units:', self.le_unit)
        layout.addRow(hw)
        layout.addRow('Motor values:', self.te_vals)
        layout.addWidget(buttonBox)
        layout.addWidget(self.info)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        for button in buttonBox.buttons():
            button.clearFocus()
            if button.text() == 'Save':
                button.clicked.connect(self.save_motor)
            elif button.text() == 'Close':
                button.clicked.connect(self.accept)

        self.cb_motor.addItems(self.MOTOR_NAMES)
        self.cb_motor.currentTextChanged.connect(self.change_motor)
        if sel_motor != '':
            self.cb_motor.setCurrentText(sel_motor)
        self.btn_file.clicked.connect(self.load_file)

    def sizeHint(self):
        return QSize(400, 50)

    def save_motor(self):
        """
        Save changes made to motor values.
        """
        self.info.setText('')
        name = self.cb_motor.currentText()
        if name != '':
            vals = [float(x) for x in self.te_vals.toPlainText().split(',')]
            if len(vals) in self.data_shape:
                axis = self.data_shape.index(len(vals))
                self.data.update_motor(name, vals, self.le_unit.text(), axis=axis)
                self.info.setText('Saved changes successfully')
                self.accept()
            else:
                self.info.setText('Length of the array {} is not matching any of data axis shape {}'.format(len(vals), self.data_shape))

    def change_motor(self, text):
        """
        Callback when selected motor is changed.

        :param text: Motor name
        :type text: str
        """
        self.le_unit.setText('')
        self.te_vals.setText('')
        if text != '':
            for m in self.motors:
                if m['name'] == text:
                    self.le_unit.setText(m['unit'])
                    if isinstance(m['values'], (list, np.ndarray)):
                        self.te_vals.setText(',\t'.join(['{:.4f}'.format(x) for x in m['values']]))
                        self.step_m.setText('{:.4f}'.format(np.mean(np.diff(m['values']))))
                        self.start_m.setText('{:.4f}'.format(m['values'][0]))
                    else:
                        self.te_vals.setText('{}'.format(m['values']))

    def change_start(self):
        vals = np.array([float(x) for x in self.te_vals.toPlainText().split(',')])
        vals = vals - vals[0] + float(self.start_m.text())
        self.te_vals.setText(',\t'.join(['{:.4f}'.format(x) for x in vals]))

    def change_step(self):
        vals = np.array([float(x) for x in self.te_vals.toPlainText().split(',')])
        vals = np.arange(len(vals)) * float(self.step_m.text()) + float(self.start_m.text())
        self.te_vals.setText(',\t'.join(['{:.4f}'.format(x) for x in vals]))

    def reload_motor(self):
        self.motors = self.motors_original
        self.motors_original = copy.deepcopy(self.motors)
        self.change_motor(self.cb_motor.currentText())

    def load_file(self):
        """
        Load motor values from a csv or txt file.
        """
        start_file = os.path.expanduser("~/")
        filt = ['.csv', '.txt']
        readers = [f for f in FileFormat.formats
                   if getattr(f, 'read', None)
                   and getattr(f, "EXTENSIONS", None)
                   and any(set(getattr(f, "EXTENSIONS", None)).intersection(filt))]
        filename, reader_nm, _ = open_filename_dialog(start_file, None, readers)
        if os.path.exists(filename):
            if reader_nm is not None:
                reader_class = class_from_qualified_name(reader_nm)
                reader = reader_class(filename)
            else:
                reader = FileFormat.get_reader(filename)
            data = reader.read()

            mdata = None
            if isinstance(data, (Table, list, tuple, np.ndarray)):
                mdata = np.array(data).ravel()
            elif isinstance(data, dict) and any(data):
                item, ok = QInputDialog.getItem(self, 'Get motor array', 'Select motor item:',
                                                [x for x in data if isinstance(data[x], np.ndarray)])
                if ok:
                    mdata = data[item].ravel()
            if np.any(mdata):
                self.te_vals.setText(',\t'.join(['{:.4f}'.format(x) for x in mdata]))


INFO_MODE = 101
PREVIEW_MODE = 102
from silx.gui import qt, icons
from silx.gui.data.DataViews import DataView
from silx.gui.widgets.TableWidget import TableWidget


class _Preview(DataView):
    """Preview subaperture data with motor positions"""

    def __init__(self, parent, dataViewer=None):
        """
        Init preview in data viewer

        :param parent: Parent object
        :type parent: QWidget
        :param dataViewer: Data viewer object
        :type dataViewer: DataViewerFrameOrange
        """
        self.dataViewer = dataViewer
        self.data = None
        super(_Preview, self).__init__(
            parent=parent,
            modeId=PREVIEW_MODE,
            label="Preview",
            icon=icons.getQIcon("eye"))

    def createWidget(self, parent):
        """
        Create a new preview plot widget.

        :param parent: Parent object
        :type parent: QWidget
        :return: Plot (2D) object
        :rtype: pylost_widgets.util.util_plots.OrangePlot2D
        """
        plot = OrangePlot2D(opacity=0.05)
        plot.getLegendsDockWidget().setVisible(True)
        return plot

    def clear(self):
        """Clear preview plot"""
        widget = self.getWidget()

    def setData(self, data):
        """
        Set the data of preview plot.

        :param data: Dataviewer data
        :type data: MetrologyData
        """
        self.data = data
        pix_size = data.pix_size_detector
        if len(pix_size) == 2:
            pix_units = [str(x.unit) for x in pix_size]
            pix_size_vals = [x.value for x in pix_size]
            lbls = ['Data Y ({})'.format(pix_units[-2]), 'Data X ({})'.format(pix_units[-1])]
            widget = self.getWidget()
            widget.clearImages()
            mx = []
            my = []
            for m in data.motors:
                if m['name'] == 'motor_X':
                    mx = m['values'] if isinstance(m['values'], (list, np.ndarray)) else [m['values']]
                if m['name'] == 'motor_Y':
                    my = m['values'] if isinstance(m['values'], (list, np.ndarray)) else [m['values']]

            for i in range(np.min([5, data.shape[0]])):
                lgnd = 'img_{}'.format(i)
                colormap = plot_win_colormap(data[i])
                origin = [0, 0]
                if len(mx) > i:
                    lgnd = '{}_X{:.1f}'.format(lgnd, mx[i])
                    origin[0] = mx[i] - (data.shape[-1] * pix_size_vals[-1] / 2)
                if len(my) > i:
                    lgnd = '{}_Y{:.1f}'.format(lgnd, my[i])
                    origin[1] = my[i] - (data.shape[-2] * pix_size_vals[-2] / 2)
                widget.addImage(data[i], legend=lgnd, colormap=colormap,
                                xlabel=lbls[-1], ylabel=lbls[-2], origin=tuple(origin),
                                scale=tuple(pix_size_vals[::-1])
                                )

    def axesNames(self, data, info):
        return None

    def getDataPriority(self, data, info):
        """
        Enable view if data is of type MetrologyData and has 3 dimensions

        :param data: Data
        :type data: np.ndarray
        :param info:
        :type info:
        :return: View suuported or not
        :rtype: int
        """
        if isinstance(data, MetrologyData) and data.ndim == 3:  # widget.isSupportedData(data):
            return 1
        else:
            return DataView.UNSUPPORTED


class _InfoView(DataView):
    """View displaying data using text"""
    rows = 10
    cols = 2

    def __init__(self, parent, dataViewer=None, editable=False, infoChanged=None):
        """
        View to display associated attributes of displayed measurement data (for MetrologyData), such as start position, detector dimensions etc. in a tabular format.

        :param parent: Parent object
        :type parent: QWidget
        :param dataViewer: Data viewer object which encapsulates various data views (raw data, image view, curve view etc.)
        :type dataViewer: DataViewerFrameOrange
        :param editable: Flag whether the corresponding data is editable
        :type editable: bool
        :param infoChanged: Signal to emit when info changed
        :type infoChanged: pyqtSignal
        """
        self.infoChanged = infoChanged
        self.editable = editable
        self.dataViewer = dataViewer
        self.data = None
        self.motor_names = {}
        super(_InfoView, self).__init__(
            parent=parent,
            modeId=INFO_MODE,
            label="Info",
            icon=icons.getQIcon("view-text"))

    def createWidget(self, parent):
        """
        Create a new table widget.

        :param parent: Parent object
        :type parent: QWidget
        :return: Table widget to show info of data
        :rtype: TableWidget
        """
        widget = TableWidget(parent)
        widget.itemChanged.connect(self.editData)
        widget.itemDoubleClicked.connect(self.dblClick)
        return widget

    def clear(self):
        """Clear info widget"""
        widget = self.getWidget()
        # widget.setRowCount(0)

    def editData(self, item):
        """
        Edit data in table items. (e.g. change start position or pixel size)

        :param item: Clicked table item
        :type item: QTableWidgetItem
        """
        row = item.row()
        col = item.column()
        widget = self.getWidget()
        if col != 1:
            return
        if row == 0:  # unit
            punit = str(self.data.unit)
            try:
                # TODO: This is workaround to keep data in the same instance after applying unit conversion. Need to improve
                np.multiply(self.data, self.data.unit.to(item.text()), out=self.data)
                self.data._set_unit(item.text())
            except Exception as e:
                print(e)
                alertMsg('Error', repr(e))
                widget.blockSignals(True)
                item.setText(punit)
                widget.blockSignals(False)
        if row == 1:  # axis names
            self.data._set_axis_names(self._parse_str(item.text()))
        if row == 2:  # Detector dimensions
            self.data._set_dim_detector(self._parse_data(item.text()))
        if row == 3:  # Pixel size
            self.data._set_pix_size(self._parse_quantity(item.text()))
        if row == 4:  # Start position
            updated = self.data._set_start_position(self._parse_quantity(item.text()))
            if not updated:
                widget.blockSignals(True)
                item.setText(self._format_vis(self.data.start_position))
                widget.blockSignals(False)
        if row == 5:  # Initial shape
            self.data._set_init_shape(self._parse_data(item.text()))
        if row == 9:  # Axis values
            self.data._set_axis_values(self._parse_axis_vals(item.text()))

        self.dataViewer.setData(self.data)
        self.infoChanged.emit()

    def dblClick(self, item):
        """
        Double click on motors item, which opens motor value editor dialog

        :param item: Clicked item
        :type item: QTableWidgetItem
        """
        row = item.row()
        # col = item.column()
        if row >= self.rows:  # Motor values
            dialog = MotorDialog(sel_motor=self.motor_names[row], data=self.data)
            if dialog.exec_():
                self.dataViewer.setData(self.data)
                self.infoChanged.emit()

    def setData(self, data):
        """
        Set the data of info view table, with various attributes of measurement data

        :param data: Dataviewer data
        :type data: MetrologyData
        """
        self.motor_names = {}
        self.data = data
        rows = self.rows
        cols = self.cols
        mrows = 1 + len(data.motors) if np.any(data.motors) else 1
        widget = self.getWidget()
        widget.blockSignals(True)
        widget.setRowCount(rows + mrows)
        widget.setColumnCount(cols)
        header = widget.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        table_items = ['Units', 'Axis names', 'Detector dimensions', 'Pixel size', 'Start position', 'Full size (pix)',
                       'Full size', 'Size (pix)', 'Size', 'Axis values']
        table_values = ['{}'.format(data.unit),
                        self._format_vis(data.axis_names),
                        self._format_vis(data.dim_detector),
                        self._format_vis(data.pix_size_detector, fmt='{:.6f}'),
                        self._format_vis(data.start_position, fmt='{:.4f}'),
                        self._format_vis(data.init_shape),
                        self._format_vis(data.size_init_detector, fmt='{:.4f}'),
                        self._format_vis(data.shape),
                        self._format_vis(data.size_detector, fmt='{:.4f}'),
                        self._format_vis(data.axis_values)]
        try:
            self.motor_names[len(table_items)] = ''
            table_items += ['Motors']
            table_values += ['']
            if np.any(data.motors):
                for m in data.motors:
                    self.motor_names[len(table_items)] = m['name']
                    table_items += [m['name']]
                    table_values += [np.array_str(np.asarray(m['values']), precision=2) + ' ' + str(m['unit'])]
        except Exception:
            print('Failed to load motor information')

        for i, key in enumerate(table_items):
            item = QTableWidgetItem(key)
            item.setFlags(Qt.ItemIsEnabled)
            val = QTableWidgetItem(table_values[i])
            val.setToolTip(table_values[i])
            if self.editable:
                if i in [0, 1, 2, 3, 4, 5]:
                    val.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
            else:
                val.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            widget.setItem(i, 0, item)
            widget.setItem(i, 1, val)

        widget.blockSignals(False)

    def axesNames(self, data, info):
        return None

    def getDataPriority(self, data, info):
        """
        Enable view if data is of type MetrologyData and has 3 dimensions

        :param data: Data
        :type data: np.ndarray
        :param info:
        :type info:
        :return: View suuported or not
        :rtype: int
        """
        # widget = self.getWidget()
        if isinstance(data, MetrologyData):  # widget.isSupportedData(data):
            return 1
        else:
            return DataView.UNSUPPORTED

    @staticmethod
    def _format_vis(val, fmt='{}'):
        """
        Format visualization text

        :param val: Values of various python data types
        :type val: tuple/list/np.ndarray
        :param fmt: Visualization format (e.g. show 3 floating points after decimal)
        :return: Formatted string of val
        :rtype: str
        """
        if val is None:
            return ''
        elif isinstance(val, (tuple, list, np.ndarray)):
            return ' x '.join(fmt.format(x) for x in val)
        else:
            return fmt.format(val)

    @staticmethod
    def _parse_axis_vals(val):
        """
        Parse axis values string. It can be a quantity array or link to motor values

        :param val: String value
        :type val: str
        :return: Axis values
        :rtype: list
        """
        try:
            if ' x ' in val:
                ret = []
                for x in val.split(' x '):
                    try:
                        ret.append(Quantity(x))
                    except Exception:
                        ret.append(x)
                return ret
            elif val.strip() == '':
                return None
            else:
                try:
                    return [Quantity(val)]
                except Exception:
                    return [val]
        except Exception:
            return None

    @staticmethod
    def _parse_str(val):
        """
        Parse string, with character ' x ' used for splitting string into array

        :param val: String value
        :type val: str
        :return: Formatted string value
        :rtype: str
        """
        try:
            if ' x ' in val:
                return [x.strip() for x in val.split(' x ')]
            elif val.strip() == '':
                return None
            else:
                return val
        except Exception:
            return None

    @staticmethod
    def _parse_data(val):
        """
        Parse string and convert to numerical/bool (if possible). The string is split at character ' x ' if present

        :param val: String value
        :type val: str
        :return: Parsed value
        :rtype: int/float/str
        """
        import ast
        try:
            if ' x ' in val:
                return [ast.literal_eval(x.strip()) for x in val.split(' x ')]
            else:
                return ast.literal_eval(val)
        except Exception:
            return None

    @staticmethod
    def _parse_quantity(val):
        """
        Parse string and convert value to astropy Quantity.

        :param val: String to parse
        :type val: str
        :return: Parsed value
        :rtype: Quantity
        """
        try:
            if ' x ' in val:
                return [Quantity(x.strip()) for x in val.split(' x ')]
            elif val.strip() == '':
                return None
            else:
                return Quantity(val)
        except Exception:
            return None
