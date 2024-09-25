# coding=utf-8
import copy
import re

import numpy as np
from Orange.widgets import gui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog, QLabel, \
    QLineEdit, QSizePolicy as Policy, QSlider, QTabWidget, QTableWidget, QTableWidgetItem, QWidget
from astropy.units import Quantity

from pylost_widgets.util import math
from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.util.math import pv, rms
from pylost_widgets.util.util_functions import DEFAULT_DATA_NAMES, copy_items, has_key_in_dict, plot_win_colormap
from pylost_widgets.util.util_plots import OrangePlot1D, OrangePlot2D
from pylost_widgets.widgets._PylostBase import PylostBase


class StatisticsTab:
    """
    Statistics tab with values like rms, pv, radius etc.
    """

    __TABS_DICT = {'compare': 'Compare datasets', 'noise': 'Noise of datasets'}
    __TABS = list(__TABS_DICT.keys())

    def __init__(self):
        super().__init__()
        self.__flag_stats_update = True

    def get_stats_update_flag(self):
        return self.__flag_stats_update

    def set_stats_update_flag(self, flag):
        self.__flag_stats_update = flag

    def add_stats_tab(self, name, header=[]):
        """
        Add statistics tab

        :param name: Tab name
        :type name: str
        :param header: Stats table header, e.g. ['height_rms', 'height_pv',...], default []
        :type header: list
        """
        self.__flag_stats_update = True
        self.tblHeader = {}
        for i, tab_name in enumerate(self.__TABS):
            self.tblHeader[tab_name] = header.copy()
            if tab_name == 'compare':
                self.tblHeader[tab_name] += ['Size']
            for key in self.DATA_NAMES:
                if key in self.DEFAULT_DATA_NAMES:
                    self.tblHeader[tab_name] += [key + '_rms', key + '_pv']
                if key == 'height' and 'slopes_x' not in self.DATA_NAMES:
                    self.tblHeader[tab_name] += ['slopes_x_rms', 'slopes_y_rms']
                elif key == 'slopes_y' and 'height' not in self.DATA_NAMES:
                    self.tblHeader[tab_name] += ['height_rms']

                if tab_name == 'compare':
                    for ext in ['_err_val', '_p', '_q', '_theta', '_radius', '_Rc_tan', '_Rc_sag']:
                        if has_key_in_dict(key + ext, self.data_in):
                            self.tblHeader[tab_name] += [key + ext]

        widget = QWidget()
        layout = QGridLayout()
        widget.setLayout(layout)
        vbox = gui.vBox(None)
        gui.checkBox(vbox, self, 'show_ellipse', 'Show ellipse parameters', callback=self.checkbox_change)
        hbox = gui.hBox(vbox, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.checkBox(hbox, self, 'show_slopes_x', 'Show slopes_x parameters', callback=self.checkbox_change)
        gui.checkBox(hbox, self, 'show_slopes_y', 'Show slopes_y parameters', callback=self.checkbox_change)
        gui.checkBox(hbox, self, 'show_height', 'Show height parameters', callback=self.checkbox_change)
        hbox = gui.hBox(vbox, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.checkBox(hbox, self, 'show_rms', 'Show rms parameters', callback=self.checkbox_change)
        gui.checkBox(hbox, self, 'show_pv', 'Show pv parameters', callback=self.checkbox_change)
        gui.checkBox(hbox, self, 'show_rc', 'Show radius parameters', callback=self.checkbox_change)
        hbox = gui.hBox(vbox, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.checkBox(hbox, self, 'show_line_stats', 'Show line statistics', callback=self.update_statistics_tabs)
        layout.addWidget(vbox, 0, 0, 1, 2)

        self.tblStats = {}
        i = 0
        for i, tab_name in enumerate(self.__TABS):
            lbl = gui.widgetLabel(None, '{} statistics'.format(self.__TABS_DICT[tab_name]))
            layout.addWidget(lbl, i * 2 + 1, 0, 1, 2)
            rows = len(self.data_in)
            if tab_name == 'noise':
                rows += 1
            self.tblStats[tab_name] = QTableWidget()
            self.tblStats[tab_name].setRowCount(rows)
            self.tblStats[tab_name].setColumnCount(len(self.tblHeader[tab_name]))
            layout.addWidget(self.tblStats[tab_name], i * 2 + 2, 0, 1, 3)
            self.tblStats[tab_name].setHorizontalHeaderLabels(self.tblHeader[tab_name])

        self.main_tabs.addTab(widget, name)

    def checkbox_change(self):
        """
        Selection of parameters to display, e.g. show_height to show only height parameters, show_rms show only rms of different data
        """
        for i, tab_name in enumerate(self.__TABS):
            for j, item in enumerate(self.tblHeader[tab_name]):
                self.tblStats[tab_name].setColumnHidden(j, False)
                if item.endswith('_p') or item.endswith('_q') or item.endswith('_theta'):
                    self.tblStats[tab_name].setColumnHidden(j, not self.show_ellipse)
                elif item.endswith('_rms'):
                    self.tblStats[tab_name].setColumnHidden(j, not self.show_rms)
                elif item.endswith('_pv'):
                    self.tblStats[tab_name].setColumnHidden(j, not self.show_pv)
                elif item.endswith('_radius') or item.endswith('_Rc_tan') or item.endswith('_Rc_sag'):
                    self.tblStats[tab_name].setColumnHidden(j, not self.show_rc)

                if item.startswith('slopes_x') and not self.show_slopes_x:
                    self.tblStats[tab_name].setColumnHidden(j, True)
                if item.startswith('slopes_y') and not self.show_slopes_y:
                    self.tblStats[tab_name].setColumnHidden(j, True)
                if item.startswith('height') and not self.show_height:
                    self.tblStats[tab_name].setColumnHidden(j, True)

    def update_statistics(self, scan, multiple=True):
        """
        Update stats from a new scan data

        :param scan: Scan data
        :type scan: dict
        :param multiple: Scan data from multiple input channels, default True
        :type multiple: bool
        """
        try:
            if multiple:
                if self.__flag_stats_update:
                    for tab_name in self.__TABS:
                        for i, key in enumerate(scan):
                            scan_data = scan[key]
                            name = self.id_name_map[key] if key in self.id_name_map else key
                            for j, item in enumerate(self.tblHeader[tab_name]):
                                if item == 'Id':
                                    self.tblStats[tab_name].setItem(i, j, QTableWidgetItem('{}'.format(name)))
                                elif item == 'Algorithm':
                                    if 'creator' in self.data_in[key]:  # ['stitch_data']:
                                        creator = self.data_in[key]['creator']  # ['stitch_data']['creator']
                                        self.tblStats[tab_name].setItem(i, j, QTableWidgetItem('{}'.format(creator)))
                                elif item == 'Size':
                                    try:
                                        for item_def in DEFAULT_DATA_NAMES:
                                            if item_def in scan_data:
                                                size = scan_data[item_def].size_detector if isinstance(
                                                    scan_data[item_def], MetrologyData) else []
                                                s = [q.value if isinstance(q, Quantity) else q for q in size]  # remove AstropyDeprecationWarning
                                                if not any(s):
                                                    vals = scan_data[item_def].get_axis_val_items_detector()
                                                    if any(vals):
                                                        size = [pv(x) for x in vals]
                                                    else:
                                                        size = scan_data[item_def].shape
                                                s = [q.value if isinstance(q, Quantity) else q for q in s]  # remove AstropyDeprecationWarning
                                                if size is not None and any(s):
                                                    sz_fmt = self._format_vis(size, fmt='{:.2f}')
                                                    self.tblStats[tab_name].setItem(i, j, QTableWidgetItem(sz_fmt))
                                                break
                                    except Exception as e:
                                        print(e)
                                # elif item.endswith('_rms') and item.split('_rms')[0] in scan_data:
                                #     self.tblStats[tab_name].setItem(i, j, QTableWidgetItem('{:.4f}'.format(rms(scan_data[item.split('_rms')[0]]))))
                                # elif item.endswith('_pv') and item.split('_pv')[0] in scan_data:
                                #     self.tblStats[tab_name].setItem(i, j, QTableWidgetItem('{:.4f}'.format(pv(scan_data[item.split('_pv')[0]]))))
                                elif item == 'Applied fit':
                                    cmt = scan_data.get('comment_log', None)
                                    if cmt is not None:
                                        index = cmt.find('Applied fit: ')
                                        if index > -1:
                                            fit_array = cmt[index+13:].split('\n')[0].split()
                                            fit_type = fit_array[0]
                                            if 'polynomial' in fit_array:
                                                fit_type = fit_array[1] + ' ' + fit_array[3] + ' ' + fit_array[4]
                                            if 'terms' in fit_type:
                                                fit_array = cmt[index + 13:].split(']')[0].split('[')
                                                fit_type = fit_array[0] + ': ' + fit_array[1].replace('\'', '')
                                            line = i
                                            if 'noise' in tab_name:
                                                line = i + 1
                                            self.tblStats[tab_name].setItem(line, j, QTableWidgetItem(fit_type))
                                elif item in scan_data:
                                    val = scan_data[item]
                                    fmt = '{:.2f}'
                                    if item.endswith(('_p',)):
                                        fmt = '{:.5f}'
                                    elif item.endswith('_q'):
                                        val = scan_data[item].to('mm')
                                    self.tblStats[tab_name].setItem(i, j, QTableWidgetItem(fmt.format(val)))
                    self.update_statistics_tabs()
                    self.__flag_stats_update = False
            self.checkbox_change()
        except Exception as e:
            self.Error.unknown(repr(e))

    def update_statistics_tabs(self):
        for tab_name in self.__TABS:
            if tab_name in self.image_data:
                tab_data = self.image_data[tab_name]
                is_2d = True
                if not any(tab_data) and tab_name in self.line_data:
                    is_2d = False
                    tab_data = self.line_data[tab_name]
                for i, legend in enumerate(tab_data):
                    data = tab_data[legend]
                    if is_2d and tab_name in self.line_data and legend in self.line_data[tab_name]:
                        data_line = self.line_data[tab_name][legend]
                    else:
                        data_line = {}
                    for j, item in enumerate(self.tblHeader[tab_name]):
                        if item == 'Id':
                            txt = legend + '\n--line' if self.show_line_stats else legend
                            self.tblStats[tab_name].setItem(i, j, QTableWidgetItem(txt))
                        else:
                            for ext in ['_rms', '_pv']:
                                k = item.split(ext)[0]
                                if item.endswith(ext) and k in data:
                                    val = rms(data[k]) if ext == '_rms' else pv(data[k])
                                    val_line = 0
                                    if k in data_line:
                                        val_line = rms(data_line[k]) if ext == '_rms' else pv(data_line[k])
                                    txt = '{:.2f} \n{:.2f}'.format(val,
                                                                   val_line) if self.show_line_stats else '{:.2f}'.format(
                                        val)
                                    self.tblStats[tab_name].setItem(i, j, QTableWidgetItem(txt))
                self.tblStats[tab_name].resizeRowsToContents()

    @staticmethod
    def _format_vis(val, fmt='{}'):
        """
        Format visualization text

        :param val: Values of various python data types
        :type val: list / numpy ndarray / float /...
        :param fmt: Visualization format (e.g. show 3 floating points after decimal)
        :type fmt: str
        :return: Formatted string
        :rtype: str

        """
        if val is None:
            return ''
        elif isinstance(val, (tuple, list, np.ndarray)):
            return ' x '.join(fmt.format(x) for x in val)
        else:
            return fmt.format(val)


class CompareTabBase:
    """
    Base class for comparison tabs of data, data noise
    """

    def __init__(self):
        super().__init__()
        self.units = {}
        self.pix_sz = {}
        self.load_stats_1d = {}
        self.load_stats_2d = {}

    def check_show(self, plots):
        """
        Show / hide plots in the comparison tabs

        :param plots: Plots 1D / 2D of different subtabs (height, slopes_x etc)
        :type plots: dict
        """
        for tab in plots:
            if self.show_2d:
                plots[tab][0].show()
            else:
                plots[tab][0].hide()
            if self.show_1d:
                plots[tab][1].show()
                plots[tab][2].show()
            else:
                plots[tab][1].hide()
                plots[tab][2].hide()

    def check_yorigin(self, plots):
        for tab in plots:
            curves = plots[tab][1].getAllCurves()
            for curve in curves:
                x, y, xerr, yerr = curve.getData()
                ynew = y - np.nanmean(y)
                curve.setData(x, ynew, xerr, yerr)
            plots[tab][1].resetZoom()

    def check_xorigin(self, plots):
        for tab in plots:
            imgs = plots[tab][0].getAllImages()
            for img in imgs:
                xmin, xmax, ymin, ymax = img._getBounds()
                xmin_new = 0 if self.x_origin else xmin - (xmin + xmax) / 2
                img.setOrigin((xmin_new, ymin))
            plots[tab][0].resetZoom()
            curves = plots[tab][1].getAllCurves()
            for curve in curves:
                x, y, xerr, yerr = curve.getData()
                xnew = x - x[0] if self.x_origin else x - np.nanmean(x)
                curve.setData(xnew, y, xerr, yerr)
            plots[tab][1].resetZoom()

    def check_normalize(self, plots):
        for tab in plots:
            imgs = plots[tab][0].getAllImages()
            for img in imgs:
                data = img.getData(copy=False)
                info = img.getInfo(copy=False)
                if not isinstance(info, dict):
                    info = {}
                zscale = 1 / math.pv(data) if self.normalize else info.get('zscale', 1)
                img.setData(data * zscale)
                info['zscale'] = math.pv(data) if self.normalize else 1
                img.setInfo(info)
            plots[tab][0].resetZoom()
            curves = plots[tab][1].getAllCurves()
            for curve in curves:
                x, y, xerr, yerr = curve.getData(copy=False)
                info = curve.getInfo(copy=False)
                if not isinstance(info, dict):
                    info = {}
                zscale = 1 / math.pv(y) if self.normalize else info.get('zscale', 1)
                curve.setData(x, y * zscale, xerr, yerr)
                info['zscale'] = math.pv(y) if self.normalize else 1
                curve.setInfo(info)
            plots[tab][1].resetZoom()
            self.update_plot_info(plots[tab])

    def add_plot_view_checks(self, vbox, plots):
        """
        Add box with 'show 2d', 'show 1d' checkboxes

        :param vbox: Parent box
        :type vbox: QWidget
        """
        box = gui.hBox(vbox, "", sizePolicy=(Policy.Fixed, Policy.Fixed), stretch=1, margin=10)
        gui.checkBox(box, self, 'show_2d', 'Show 2D plots', callback=lambda: self.check_show(plots), labelWidth=150)
        gui.checkBox(box, self, 'show_1d', 'Show 1D plots', callback=lambda: self.check_show(plots), labelWidth=150)
        gui.checkBox(box, self, 'x_origin', 'Start X at origin', callback=lambda: self.check_xorigin(plots),
                     labelWidth=150)
        gui.checkBox(box, self, 'normalize', 'Normalize data', callback=lambda: self.check_normalize(plots),
                     labelWidth=150)
        gui.checkBox(box, self, 'level_data', 'All curves are levelled', callback=lambda: self.check_yorigin(plots),
                     labelWidth=150)
        box.setAutoFillBackground(True)
        p = box.palette()
        p.setColor(box.backgroundRole(), Qt.lightGray)
        box.setPalette(p)

    @staticmethod
    def clear_base_tab_widget(widget):
        """
        Clear compare tab widget elements, plot 2d, plot1d, slider

        :param widget: Tab widget with plots
        :type widget: QWidget
        """
        child = widget.findChild(OrangePlot2D)
        if child is not None:
            child.clear()
        child = widget.findChild(OrangePlot1D)
        if child is not None:
            child.clear()
        widget.findChild(QSlider).setMaximum(0)

    def _add_to_stack(self, widget, stack):
        """
        Add removed widget to a stack

        :param widget: Tab widget
        :type widget: QWidget
        :param stack: Stack of tab widgets
        :type stack: list
        """
        try:
            self.clear_base_tab_widget(widget)
            stack.append(widget)
        except Exception as e:
            print(e)

    def init_default_stack(self, stack, cnt):
        if len(stack) == 0:
            for i in range(cnt):
                widget, widget_items = self._get_new_subtab(show_legend=False)
                stack.append(widget)

    def add_base_tab(self, name, tabs, plots, stack, load_contents=True, callback_slider=None):
        vbox = gui.vBox(None)
        vbox.layout().addWidget(tabs)
        self.add_plot_view_checks(vbox, plots)
        self.main_tabs.addTab(vbox, name)

        for i in np.arange(tabs.count())[::-1]:
            self._add_to_stack(tabs.widget(i), stack)
            tabs.removeTab(i)
        if load_contents:
            self.load_tab_contents(tabs, plots, stack, callback_slider)

    def is_tab_initialized(self, tabs):
        current_tab_names = [tabs.tabText(i) for i in np.arange(tabs.count())]
        for item in self.DATA_NAMES:
            if item in current_tab_names:
                return True
        return False

    def load_tab_contents(self, tabs, plots, stack, callback_slider):
        current_tab_names = [tabs.tabText(i) for i in np.arange(tabs.count())]
        flag = False
        for item in self.DATA_NAMES:
            if item not in current_tab_names:
                flag = True
                if len(stack) > 0:
                    widget = stack.pop(0)
                    tabs.addTab(widget, item)
                    plots[item] = [widget.findChild(OrangePlot2D), widget.findChild(OrangePlot1D),
                                   widget.findChild(QSlider), widget.findChild(QLabel)]
                else:
                    widget, widget_items = self._get_new_subtab()
                    tabs.addTab(widget, item)
                    plots[item] = widget_items
                    widget_items[2].valueChanged.connect(lambda nm=item: callback_slider(nm))
        return flag

    def _get_new_subtab(self, show_legend=True):
        """
        Create a new comparison tab

        :return: New widget, [2d plot, 1d plot, slider]
        :rtype: QWidget, [CustomPlot2D, CustomPlot1D, QSlider]
        """
        widget = QWidget()
        layout = QGridLayout()
        widget.setLayout(layout)
        pt2 = OrangePlot2D(show_legend=show_legend)
        pt1 = OrangePlot1D(show_legend=show_legend)
        sld = QSlider(Qt.Horizontal)
        info = QLabel()
        layout.addWidget(pt2, 0, 0)
        layout.addWidget(pt1, 1, 0)
        layout.addWidget(info, 2, 0)
        layout.addWidget(sld, 3, 0)
        pt1.getLegendsDockWidget().sigCommonEvents.connect(self.common_legend_events)
        pt2.getLegendsDockWidget().sigCommonEvents.connect(self.common_legend_events)

        return widget, [pt2, pt1, sld, info]

    def common_legend_events(self, cdict):
        pass

    @staticmethod
    def addPlotStats(plot_win, row=2, items=[], fix_len=True):
        """
        Add additional parameters to display in the traditional live X/Y/Data display widget, such as data rms, pv, std and dimensions
        :param positionInfo: Widget containing position information
        :return:
        """
        pwLayout = plot_win.centralWidget().layout()
        if type(pwLayout) is QGridLayout:
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            for name, val in items:
                layout.addWidget(QLabel('<b>' + name + ':</b>'))
                contentWidget = QLabel()
                if len(val.split('|')) > 1:
                    contentWidget.setText(val.split('|')[0] + ',...')
                else:
                    contentWidget.setText(val)
                contentWidget.setTextInteractionFlags(Qt.TextSelectableByMouse)
                if fix_len:
                    contentWidget.setFixedWidth(contentWidget.fontMetrics().boundingRect('############').width())
                contentWidget.setToolTip(val)
                layout.addWidget(contentWidget)

            layout.addStretch(1)
            bottomStats = QWidget(None)
            bottomStats.setLayout(layout)
            w = pwLayout.itemAtPosition(row, 0)
            if w is not None:
                w.widget().setParent(None)
                # w.widget().deleteLater()
            pwLayout.addWidget(bottomStats, row, 0, 1, -1)

    @staticmethod
    def convert_item(data, unit):
        try:
            data = data.to(unit)  # convert to already existing plot units
        except Exception as e:
            raise Exception(
                'Unable to convert to same units; unit_data1={}, unit_data2={}, Error={}'.format(unit, str(data.unit),
                                                                                                 e))
        return data

    @staticmethod
    def colormap_changed(item, plots, active_legend):
        """
        Colormap values are changed and updated in image plots. Changes from one image are applied to all images in the plot2d viewer.
        :param self:
        :param item: Viewer name (e.g. slopes_x) where colormap changed
        :param active_legend: Legend of the image where colormap changed
        :return:
        """
        try:
            image = plots[item][0].getImage(legend=active_legend)
            colormap = image.getColormap() if (image is not None) else None
            if colormap is not None:
                legends = plots[item][0].getAllImages(just_legend=True)
                for x in legends:
                    if x != active_legend:
                        image = plots[item][0].getImage(legend=x)
                        if image is not None:
                            image.setColormap(colormap)
        except Exception as e:
            print(e)

    @staticmethod
    def get_binY(img):
        return 1 + (img.getData(copy=False).shape[0] // 500)

    @staticmethod
    def get_binX(img):
        return 1 + (img.getData(copy=False).shape[1] // 500)

    def merge_colormaps(self, plots):
        try:
            for item in self.DATA_NAMES:
                imgs = plots[item][0].getAllImages()
                if imgs is None:
                    return
                if len(imgs) > 0:
                    all_data_pixels = np.concatenate(tuple(
                        img.getData(copy=False)[::self.get_binY(img), ::self.get_binX(img)].ravel() for img in imgs))
                    cm = 'Greys_r' if 'slope' in item else 'turbo'
                    colormap = plot_win_colormap(all_data_pixels, cmap_name=cm)
                    for img in imgs:
                        img.setColormap(colormap)
                    # vmin = 0
                    # vmax = 0
                    # for img in imgs:
                    #     cmap = img.getColormap()
                    #     vmin = np.min([vmin, cmap['vmin']]) if vmin!=0 and cmap['vmin'] is not None else cmap['vmin']
                    #     vmax = np.max([vmax, cmap['vmax']]) if vmax!=0 and cmap['vmax'] is not None else cmap['vmax']
                    # for img in imgs:
                    #     cmap = img.getColormap()
                    #     cmap_new = Colormap(name=cmap['name'], vmin=vmin, vmax=vmax)
                    #     cmap_new.sigChanged.connect(lambda item=item, legend=img.getName(): self.colormap_changed(item, plots, active_legend=legend))
                    #     img.setColormap(cmap_new)
        except Exception as e:
            print('merge_colormaps<-CompareTabBase')
            print(e)

    def validate_units(self, item, data):
        if data.ndim > 2 or data.ndim == 0:
            raise Exception(
                'Data must be either 1D or 2D. This widget cannot be used for {} dimensional data'.format(data.ndim))
        if item not in self.units:
            self.units[item] = []
        if not any(self.units[item]):  # This is the first legend
            if isinstance(data, MetrologyData):
                self.units[item] = [str(data.unit)]
                if np.any(data.dim_detector):
                    self.units[item] += [str(x.unit) for x in data.pix_size_detector]
                else:
                    self.units[item] += [str(x.unit) if any(x.value) else 'pix' for x in data.get_axis_val_items()]
                if len(self.units[item]) == 2:  # i.e. 1d data
                    self.units[item].insert(1, 'pix')
            else:
                self.units[item] = ['dimensionless', 'pix', 'pix']  # data, y, x
        else:  # Atleast one legend is already added
            if isinstance(data, MetrologyData):
                if str(data.unit) != self.units[item][0]:  # Data units don't match.
                    data = self.convert_item(data, self.units[item][0])
                if np.any(data.dim_detector):
                    pix_sz_inv = data.pix_size_detector[::-1]
                    pix_sz_inv_new = copy.deepcopy(pix_sz_inv)
                    uxy = self.units[item][1:][::-1]
                    for i, x in enumerate(pix_sz_inv):
                        if str(x.unit) != uxy[i] and str(x.unit) != 'pix' and uxy[i] != 'pix':
                            pix_sz_inv_new[i] = self.convert_item(x, uxy[i])
                            data._set_pix_size(pix_sz_inv_new[::-1])
                else:
                    axValsInv = data.get_axis_val_items()[::-1]
                    axValsInvNew = copy.deepcopy(axValsInv)
                    uxy = self.units[item][1:][::-1]
                    for i, x in enumerate(axValsInv):
                        if any(x.value) and str(x.unit) != uxy[i] and str(x.unit) != 'pix' and uxy[i] != 'pix':
                            axValsInvNew[i] = self.convert_item(x, uxy[i])
                            data._set_axis_values(axValsInvNew[::-1])

        return data

    @staticmethod
    def clear_plots(plots, item):
        i = plots[item][0]
        if i is not None:
            i.clear()
        i = plots[item][1]
        if i is not None:
            i.clear()

    @staticmethod
    def init_slider(sld, limits, step, val):
        sld.blockSignals(True)
        sld.setMinimum(limits[0] / step)
        sld.setMaximum(limits[1] / step)
        sld.setTickInterval(20)
        sld.setSingleStep(1)
        sld.setValue(val / step)
        sld.blockSignals(False)

    def add_line_data(self, legend, item, x, data, units, tab_name='data'):
        if tab_name not in self.line_data:
            self.line_data[tab_name] = {}
        if legend not in self.line_data[tab_name]:
            self.line_data[tab_name][legend] = {}
        if not isinstance(data, MetrologyData):
            self.line_data[tab_name][legend][item + '_x'] = x
            self.line_data[tab_name][legend][item + '_units'] = units
        self.line_data[tab_name][legend][item] = data

    def add_image_data(self, legend, item, data, pix_sz, units, tab_name='data'):
        if tab_name not in self.image_data:
            self.image_data[tab_name] = {}
        if legend not in self.image_data[tab_name]:
            self.image_data[tab_name][legend] = {}
        if not isinstance(data, MetrologyData):
            self.image_data[tab_name][legend][item + '_pix_sz'] = pix_sz
            self.image_data[tab_name][legend][item + '_units'] = units
        self.image_data[tab_name][legend][item] = data

    def add_line_info(self, tab_name, line_pos={}, line_width=0, line_seq=[]):
        self.line_data[tab_name]['line_positions'] = line_pos
        self.line_data[tab_name]['line_width_mm'] = line_width
        self.line_data[tab_name]['line_legends_sequence'] = line_seq

    @staticmethod
    def get_data_curve(data, line, line_width, pix_sz_y):
        data_curve = data.value[line, :] * data.unit if isinstance(data, MetrologyData) else data[line, :]
        if line_width > 0:
            line_width_pix = line_width / pix_sz_y
            line_st = np.max([0, line - int(line_width_pix / 2)])
            line_en = np.min([data.shape[0], line + int((line_width_pix + 1) / 2)])
            data_curve = math.nanmean(data[line_st:line_en, :], axis=0)
        return data_curve

    def change_slider(self, val, item='', plots=None, sel_lines=None, line_width=0, tab_name='data'):
        # self.line_data = {}
        val = val * self.slider_step
        if item != '' and item in sel_lines and item in plots and val != list(sel_lines[item].values())[0]:
            # sel_lines[item] = {x:val for x in sel_lines[item]}
            legends = plots[item][0].getAllImages(just_legend=True)
            ref_idx = np.argmax(np.asfarray(list(self.pix_sz[item].values()))[:, -1])
            ref_pos = 0
            if len(self.line_pos) > ref_idx:
                ref_pos = self.line_pos[ref_idx]
            for i, lg in enumerate(legends):
                pix_sz_y = self.pix_sz[item][lg][-2]
                line_pos = 0
                if len(self.line_pos) > ref_idx:
                    line_pos = self.line_pos[i]
                line = int((line_pos + val - ref_pos) / self.slider_step)
                data = plots[item][0].getImage(lg).getData(copy=False)
                if line >= -data.shape[0] / 2 and line < data.shape[0] / 2:
                    line += int(data.shape[0] / 2)
                    sel_lines[item][lg] = line_pos + val - ref_pos
                    x = np.arange(data.shape[-1]) * self.pix_sz[item][lg][-1]
                    x = x - np.nanmean(x)
                    data_curve = self.get_data_curve(data, line, line_width, pix_sz_y)
                    plots[item][1].addCurve(x, data_curve, legend=lg)
                    self.add_line_data(lg, item, x, data_curve, self.units[item], tab_name=tab_name)
            lines = ', '.join(['{:.2f}'.format(x) for x in sel_lines[item].values()])
            title = 'Line (Y) = {} ({}); width = {} ({})'.format(lines, self.units[item][-2], line_width,
                                                                 self.units[item][-2])
            plots[item][1].setGraphTitle(title)
            self.add_line_info(tab_name, line_pos=sel_lines, line_width=line_width,
                               line_seq=list(sel_lines[item].keys()))

        if np.any(self.line_data) and hasattr(self.Outputs, 'lines'):
            self.Outputs.lines.send(self.line_data)
        self.check_xorigin(plots)
        self.check_normalize(plots)
        self.check_yorigin(plots)
        if len(self.manipulations) > 0:
            for manipulation in self.manipulations:
                self.common_legend_events(manipulation)
        return sel_lines

    def update_compare_1d(self, data, item, lgnd, plots, tab_name='data'):
        self.pix_sz[item][lgnd] = data.pix_size_detector[-1] if isinstance(data, MetrologyData) and len(
            data.pix_size_detector) > 0 else 1
        x = np.arange(len(data))
        data_unit = ''
        xlabel = 'X'
        if isinstance(data, MetrologyData):
            data_unit = data.unit
            axis_vals = data.get_axis_val_items()
            if isinstance(axis_vals[-1], Quantity) and any(axis_vals[-1].value) and len(axis_vals[-1]) == len(data):
                x = axis_vals[-1].value
                xlabel = 'X ({})'.format(axis_vals[-1].unit)
            elif any(data.pix_size_detector):
                x = x * data.pix_size_detector[-1].value
                xlabel = '{} ({})'.format(data.axis_names_detector[-1], data.pix_size_detector[-1].unit)
        x = x - np.nanmean(x)
        plots[item][1].addCurve(x, data, legend=lgnd, xlabel=xlabel, ylabel='Data ({})'.format(data_unit))
        # try:
        #     if isinstance(data, MetrologyData) and data.get_flag('psd_data'):
        #         # plots[item][1].getXAxis().setScale(Axis.LOGARITHMIC)
        #         plots[item][1].getYAxis().setScale(Axis.LOGARITHMIC)
        #         plt2 = plots[item][0]
        #         imgs = plt2.getAllImages()
        #         for img in imgs:
        #             img.getColormap().setNormalization('log')
        # except Exception:
        #     pass

        self.add_line_data(lgnd, item, x, data, self.units[item], tab_name=tab_name)

    def update_compare_2d(self, data, item, lgnd, nb_inputs, idx_input, plots, sel_lines, line_pos, line_width,
                          tab_name='data'):
        lbls = ['Y', 'X']
        self.pix_sz[item][lgnd] = [1, 1]
        data_unit = ''
        if isinstance(data, MetrologyData):
            if data.has_flag('invert_y_axis') and data.get_flag('invert_y_axis'):
                plots[item][0].setYAxisInverted(True)
            else:
                plots[item][0].setYAxisInverted(False)
            data_unit = data.unit
            lbls = [x + ' ({})'.format(self.units[item][i + 1]) for i, x in enumerate(data.axis_names_detector)]
            self.pix_sz[item][lgnd] = [x.value for x in data.pix_size_detector[-2:]]

        legends = plots[item][0].getAllImages(just_legend=True)
        offset_factor = 1.1
        if lgnd in legends:
            temp = plots[item][0].getImage(lgnd).getOrigin()
            offset_y = temp[1]
        else:
            offset_y = - np.sum([offset_factor * self.pix_sz[item][x][-2] *
                                 plots[item][0].getImage(x).getData(copy=False).shape[-2] for
                                 x in legends]) - data.shape[-2] * self.pix_sz[item][lgnd][-2]
        offset_x = 0 - data.shape[-1] * self.pix_sz[item][lgnd][-1] / 2
        colormap = None  # plot_win_colormap(data)

        plots[item][0].setGraphTitle('{} ({})'.format(item, data_unit))
        plots[item][0].addImage(data, legend=lgnd, info={}, selectable=True,
                                colormap=colormap, xlabel=lbls[-1], ylabel=lbls[-2], origin=(offset_x, offset_y),
                                scale=tuple(self.pix_sz[item][lgnd][::-1]),
                                copy=False)
        x = np.arange(data.shape[-1]) * self.pix_sz[item][lgnd][-1]
        x = x - np.nanmean(x)
        line = int(data.shape[-2] / 2)
        sel_lines[item][lgnd] = 0
        if nb_inputs == len(line_pos):
            new_line = int(np.round(line_pos[idx_input] / self.pix_sz[item][lgnd][-2]))
            if new_line >= -data.shape[0] / 2 and new_line < data.shape[0] / 2:
                line += new_line
                sel_lines[item][lgnd] = new_line * self.pix_sz[item][lgnd][-2]

        data_curve = self.get_data_curve(data, line, line_width, self.pix_sz[item][lgnd][-2])
        plots[item][1].addCurve(x, data_curve, legend=lgnd, xlabel=lbls[-1], ylabel='Data ({})'.format(data_unit),
                                copy=False)
        # if isinstance(data, MetrologyData) and data.get_flag('psd_data'):
        #     plots[item][1].getXAxis().setScale(Axis.LOGARITHMIC)
        #     plots[item][1].getYAxis().setScale(Axis.LOGARITHMIC)

        self.add_line_data(lgnd, item, x, data_curve, self.units[item], tab_name=tab_name)
        self.add_image_data(lgnd, item, data, self.pix_sz[item][lgnd], self.units[item], tab_name=tab_name)
        return sel_lines

    def update_base_tabs(self, scan, plots, sel_lines, line_pos, line_width, tab_name='compare',
                         callback_plot1d_click=None):
        self.units = {}
        self.pix_sz = {}
        for item in self.DATA_NAMES:
            nb_inputs = len(scan)
            sel_lines[item] = {}
            self.pix_sz[item] = {}
            slider_limits = [0.0, 0.0]
            self.slider_step = 10
            stats_items_2d = {'Rms': [], 'PV': [], 'Std': [], 'RealDims': []}
            stats_items_1d = {'Rms': [], 'PV': [], 'Std': []}
            self.load_stats_1d[item] = True
            self.load_stats_2d[item] = True
            self.clear_plots(plots, item)
            legends = []
            for i, key in enumerate(scan):
                if item in scan[key] and isinstance(scan[key][item], np.ndarray):
                    data = scan[key][item]
                    lgnd = '{}'.format(self.id_name_map[key])
                    legends += lgnd
                    data = self.validate_units(item, data)
                    if data.ndim == 1:
                        self.update_compare_1d(data, item, lgnd, plots=plots, tab_name=tab_name)
                        sel_lines[item][lgnd] = 0
                    elif data.ndim == 2:
                        sel_lines = self.update_compare_2d(data, item, lgnd, nb_inputs, i, plots=plots,
                                                           sel_lines=sel_lines, line_pos=line_pos,
                                                           line_width=line_width, tab_name=tab_name)
                        self.slider_step = max(np.asfarray(list(self.pix_sz[item].values()))[:, -1])
                        slider_limits = [np.min([slider_limits[0], -data.shape[-2] * self.slider_step / 2]),
                                         np.max([slider_limits[1], data.shape[-2] * self.slider_step / 2])]
                    fig1d = plots[item][1]._backend.fig
                    fig2d = plots[item][0]._backend.fig
                    fig1d.canvas.mpl_connect('button_press_event', callback_plot1d_click)
                    fig1d.canvas.mpl_connect('motion_notify_event',
                                             lambda event, plt=plots[item][1], item=item: self.onhover_plot1d(event,
                                                                                                              plt,
                                                                                                              item))
                    fig2d.canvas.mpl_connect('motion_notify_event',
                                             lambda event, plt=plots[item][0], item=item: self.onhover_plot2d(event,
                                                                                                              plt,
                                                                                                              item))
                    fig2d.canvas.mpl_connect('button_press_event',
                                             lambda event, plt=plots[item][0], item=item: self.onclick_plot2d(event,
                                                                                                              plt,
                                                                                                              item))
            self.add_line_info(tab_name, line_pos=sel_lines, line_width=line_width, line_seq=legends)

            title = 'Line (Y) = {} ({}); width = {} ({})'.format(
                ', '.join(['{:.2f}'.format(x) for x in sel_lines[item].values()]),
                self.units[item][-2], line_width, self.units[item][-2])
            plots[item][1].setGraphTitle(title)
            self.addPlotStats(plots[item][0], row=2, items=[(x, ' | '.join(stats_items_2d[x])) for x in stats_items_2d])
            self.addPlotStats(plots[item][1], row=2, items=[(x, ' | '.join(stats_items_1d[x])) for x in stats_items_1d])
            if self.show_2d:
                plots[item][0].show()
            if self.show_1d:
                plots[item][1].show()
            self.init_slider(plots[item][2], slider_limits, self.slider_step, list(sel_lines[item].values())[0])

        self.merge_colormaps(plots)
        self.check_normalize(plots)
        self.check_xorigin(plots)
        self.check_yorigin(plots)
        if np.any(self.line_data) and hasattr(self.Outputs, 'lines'):
            self.Outputs.lines.send(self.line_data)

    def onclick_plot2d(self, event, plt, item):
        try:
            if item != '':
                plt.getLegendsDockWidget().update_alpha_all()
        except Exception as e:
            self.Error.unknown(str(e))

    def onclick_plt1d(self, event, plt, item):
        try:
            if event.dblclick and item != '':
                title = plt.getGraphTitle()
                # vals = [x.split('=')[1].strip() for x in title.split(';')]
                vals = re.split('\=|\(|\)|\;', title)
                vals = [vals[3], vals[4], vals[7], vals[8]]
                dialog = EditTitle(vals=vals)
                if dialog.exec_():
                    new_vals = dialog.getInputs()
                    self.line_pos = [float(x) for x in new_vals[0].strip().split(',')] if new_vals[0] != '' else []
                    self.line_width = float(new_vals[1].strip()) if new_vals[1] != '' else 0
                    return True, self.line_pos, self.line_width
            elif item != '':
                plt.getLegendsDockWidget().update_alpha_all()
                return False, [], 1
        except Exception as e:
            self.Error.unknown(str(e))
        return False, [], 1

    def onhover_plot1d(self, event, plt, item):
        if self.load_stats_1d[item]:
            stats = {'Rms': [], 'PV': [], 'Std': []}
            curves = plt.getAllCurves()
            for curve in curves:
                data_curve = curve.getYData(copy=False)
                stats['Rms'] += ['{:.2f}'.format(rms(data_curve))]
                stats['PV'] += ['{:.2f}'.format(pv(data_curve))]
                stats['Std'] += ['{:.2f}'.format(math.nanstd(data_curve))]
                self.addPlotStats(plt, row=2, items=[(x, ' | '.join(stats[x])) for x in stats])
            self.load_stats_1d[item] = False

    def onhover_plot2d(self, event, plt, item):
        if self.load_stats_2d[item]:
            stats = {'Rms': [], 'PV': [], 'Std': [], 'RealDims': []}
            imgs = plt.getAllImages()
            for img in imgs:
                data = img.getData(copy=False)
                stats['Rms'] += ['{:.2f}'.format(rms(data))]
                stats['PV'] += ['{:.2f}'.format(pv(data))]
                stats['Std'] += ['{:.2f}'.format(math.nanstd(data))]
                stats['RealDims'] += [
                    'x'.join(['{:.1f}'.format(x) for x in data.size_detector[::-1]]) if isinstance(data,
                                                                                                   MetrologyData) else '-']
                self.addPlotStats(plt, row=2, items=[(x, ' | '.join(stats[x])) for x in stats])
            self.load_stats_2d[item] = False


class CompareDataTab(CompareTabBase):

    def __init__(self):
        super().__init__()

        self.compare_plots = {}
        self.compare_tabs = QTabWidget(None)

        self.__stack = []

    def get_compare_stack(self):
        return self.__stack

    def init_compare_stack(self, callback, cnt=3):
        self.init_default_stack(self.__stack, cnt)

    def add_compare_tab(self, name, load_contents=True):
        self.add_base_tab(name, self.compare_tabs, self.compare_plots, self.__stack, load_contents,
                          callback_slider=self.change_compare_data_slider)

    def load_compare_tab_contents(self):
        return self.load_tab_contents(self.compare_tabs, self.compare_plots, self.__stack,
                                      self.change_compare_data_slider)

    def update_compare_tabs(self, scan):
        self.line_data['compare'] = {}
        self.image_data['compare'] = {}
        self.sel_lines = {}
        if not np.any(self.compare_plots):
            return
        self.update_base_tabs(scan, self.compare_plots, self.sel_lines, self.line_pos, self.line_width, 'compare',
                              self.onclick_compare_data_plt1d)

    def onclick_compare_data_plt1d(self, event, item=''):
        item = self.DATA_NAMES[self.compare_tabs.currentIndex()]
        flag, self.line_pos, self.line_width = self.onclick_plt1d(event, self.compare_plots[item][1], item)
        if flag:
            self.load_data(compare=True, all=False)
        return flag

    def change_compare_data_slider(self, val):
        item = self.DATA_NAMES[self.compare_tabs.currentIndex()]
        self.sel_lines = super().change_slider(val, item, plots=self.compare_plots, sel_lines=self.sel_lines,
                                               line_width=self.line_width, tab_name='compare')


class NoiseStatsTab(CompareTabBase):
    def __init__(self):
        super().__init__()

        self.noise_plots = {}
        self.noise_tabs = QTabWidget(None)

        self.__stack = []

    def add_noise_stats_tab(self, name, load_contents=True):
        self.add_base_tab(name, self.noise_tabs, self.noise_plots, self.__stack, load_contents=load_contents,
                          callback_slider=self.change_noise_slider)

    def load_noise_tab_contents(self):
        return self.load_tab_contents(self.noise_tabs, self.noise_plots, self.__stack, self.change_noise_slider)

    def get_scan_noise(self, scan):
        scan_noise = {}
        scan_noise_merge = {}
        copy_items(scan, scan_noise, deepcopy=True)

        for item in self.DATA_NAMES:
            scan_item_avg = None
            cnt = 0
            for i, key in enumerate(scan):
                if item in scan[key] and isinstance(scan[key][item], np.ndarray):
                    scan_item = scan[key][item]
                    if scan_item_avg is None:
                        scan_item_avg = scan_item
                        cnt += 1
                    else:
                        if scan_item.shape == scan_item_avg.shape:
                            scan_item_avg = scan_item_avg + scan_item
                            cnt += 1
                        else:
                            raise Exception('Cannot calculate noise. Data shapes do not match.')
            if cnt <= 1 or len(scan) == 1:
                raise Exception('Cannot calculate noise. Only one input is added.')

            scan_item_avg = scan_item_avg / cnt
            scan_noise_merge[item] = np.full((cnt,) + scan_item_avg.shape, np.nan)
            for i, key in enumerate(scan):
                if item in scan[key] and isinstance(scan[key][item], np.ndarray):
                    scan_item = scan[key][item]
                    scan_noise[key][item] = scan_item - scan_item_avg
                    scan_noise_merge[item][i, ...] = scan_item - scan_item_avg
            scan_noise_merge[item] /= np.sqrt(cnt)
            if isinstance(scan_item_avg, MetrologyData):
                scan_noise_merge[item] = scan_item_avg.copy_to(scan_noise_merge[item])
        return scan_noise, scan_noise_merge

    def update_noise_tabs(self, scan):
        self.line_data['noise'] = {}
        self.image_data['noise'] = {}
        self.sel_lines = {}
        if not np.any(self.noise_plots) or len(scan) == 1:
            return
        scan_noise, scan_noise_merge = self.get_scan_noise(scan)
        self.image_data['noise']['__full'] = scan_noise_merge
        if any(scan_noise):
            self.update_base_tabs(scan_noise, self.noise_plots, self.sel_lines, self.line_pos, self.line_width,
                                  'noise', self.onclick_noise_plt1d)

    def onclick_noise_plt1d(self, event, item=''):
        item = self.DATA_NAMES[self.noise_tabs.currentIndex()]
        flag, self.line_pos, self.line_width = self.onclick_plt1d(event, self.noise_plots[item][1], item)
        if flag:
            self.load_data(noise=True, all=False)
        return flag

    def change_noise_slider(self, val):
        item = self.DATA_NAMES[self.noise_tabs.currentIndex()]
        self.sel_lines = super().change_slider(val, item, plots=self.noise_plots, sel_lines=self.sel_lines,
                                               line_width=self.line_width, tab_name='noise')


class VisCompareBase(PylostBase, StatisticsTab, NoiseStatsTab, CompareDataTab):

    def __init__(self):
        self.line_data = {}
        self.image_data = {}
        PylostBase.__init__(self)
        NoiseStatsTab.__init__(self)
        CompareDataTab.__init__(self)
        StatisticsTab.__init__(self)
        self.data_index = []
        self.id_name_map = {}

    def init_data(self):
        self.data_in = {}
        self.id_name_map = {}
        for idx, data in enumerate(self.data_index):
            if data is not None:
                self.data_in[idx] = data
        for idx, name in enumerate(self.legend_names):
            self.id_name_map[idx] = name
        self.has_updates = True
        self.btnReload.show()

    def load_data(self, multi=False):
        super().load_data(multi)


    def init_plots(self):
        self.compare_plots = {}
        self.noise_plots = {}

    # def update_tabs(self, data_in, show_all_default_names=False, show_only_default=False, multiple=False):
    def update_tabs(self, show_all_default_names=False, show_only_default=False, multiple=True):
        self.update_data_names(self.data_in, show_only_default=show_only_default,
                               show_all_default_names=show_all_default_names, multiple=multiple)
        self.init_plots()
        for i in np.arange(self.main_tabs.count())[::-1]:
            self.main_tabs.removeTab(i)

    def load_title(self, data, id):
        text, ok = QInputDialog.getText(self, 'Legend ({})'.format(self.name), 'Enter title (legend) for the datasets:')
        name = str(text) if ok and str(text) != '' else '_{}'.format(id)
        self.legend_names.insert(id, name)

    def update_legend_info(self, plots_item, legend, legend_info):
        plt1d = plots_item[1]
        curve = plt1d.getCurve(legend)
        if curve is None:
            return
        curve_info = curve.getInfo()
        for key in legend_info:
            if key == 'offset_x':
                val = curve_info.get(key, 0)
                val += legend_info[key]['value']
                if legend_info[key]['absolute']:
                    val -= curve.getBounds()[0]
                curve_info[key] = val
        plt1d.getCurve(legend).setInfo(curve_info)
        self.update_plot_info(plots_item)

    @staticmethod
    def update_plot_info(plots_item):
        txt = ''
        plt1d = plots_item[1]
        for curve in plt1d.getAllCurves():
            curve_info = curve.getInfo()
            if np.any(curve_info):
                legend_txt = ''
                for key in curve_info:
                    if key == 'zscale' and curve_info[key] == 1:
                        continue
                    try:
                        legend_txt += ' {} = {:.4f},'.format(key, curve_info[key])
                    except:
                        legend_txt += ' {} = {},'.format(key, curve_info[key])
                if legend_txt != '':
                    if txt == '':
                        txt = 'Additional Info: '
                    txt += '{}: {}'.format(curve.getLegend(), legend_txt)
                    txt += '\t'
        plots_item[3].setText(txt)

    @staticmethod
    def reshow_legends(plt):
        plt.getLegendsDockWidget().setVisible(False)
        plt.getLegendsDockWidget().setVisible(True)

    def common_legend_events(self, cdict):
        if cdict['event'] == 'clearChanges':
            if self.is_tab_initialized(self.compare_tabs):
                self.load_data(compare=True, all=False)
            if self.is_tab_initialized(self.noise_tabs):
                self.load_data(noise=True, all=False)
            self.manipulations = []
            return
        self.create_manipulations_history(cdict)
        self.common_legend_events_2d(cdict)
        self.common_legend_events_1d(cdict)
        if cdict['event'] == 'renameCurve':
            self.rename_legend_stats(oldLegend=cdict['legend'], newLegend=cdict['newLegend'])

    def create_manipulations_history(self, cdict):
        if cdict['event'] not in ('offsetX', 'changeLineWidth', 'changeColor'):
            return
        update = False
        for manipulation in self.manipulations:
            event = manipulation['event']
            if event == cdict['event'] and manipulation['legend'] == cdict['legend']:
                update = True
                if event == 'offsetX':
                    if not manipulation.get('updated', False):
                        manipulation['offset'] += cdict['offset']
                        manipulation['updated'] = True
                if event == 'changeLineWidth':
                    manipulation['lineWidth'] = cdict['lineWidth']
                if event == 'changeColor':
                    manipulation['color'] = cdict['color']
        if not update:
            self.manipulations.append(cdict)

    def common_legend_events_1d(self, cdict):
        for item in self.DATA_NAMES:
            for plots in [self.compare_plots, self.noise_plots]:
                if item in plots:
                    plt = plots[item][1]
                    plt.blockSignals(True)
                    if cdict['event'] == 'renameCurve':
                        self.rename_legend_1d(plt, oldLegend=cdict['legend'], newLegend=cdict['newLegend'])
                        self.reshow_legends(plt)
                    if cdict['event'] == 'removeCurve':
                        self.remove_legend_1d(plt, cdict['legend'])
                        self.reshow_legends(plt)
                    if cdict['event'] == 'offsetX':
                        legend_info = {'offset_x': {'value': cdict['offset'],
                                                    'absolute': cdict['absolute']}
                                       }
                        self.update_legend_info(plots[item], cdict['legend'], legend_info)
                        plt.getLegendsDockWidget().offset_x(cdict['legend'], cdict['offset'],
                                                            absolute=cdict['absolute'])
                    if cdict['event'] == 'changeLineWidth':
                        plt.getCurve(cdict['legend']).setLineWidth(cdict['lineWidth'])
                        self.reshow_legends(plt)
                    if cdict['event'] == 'changeColor':
                        plt.getCurve(cdict['legend']).setColor(cdict['color'])
                        self.reshow_legends(plt)
                    plt.blockSignals(False)

    @staticmethod
    def rename_legend_1d(plt, oldLegend, newLegend):
        legends = plt.getAllCurves(just_legend=True)
        if oldLegend in legends:
            plt.getLegendsDockWidget().renameCurve(oldLegend, newLegend)

    @staticmethod
    def remove_legend_1d(plt, legend):
        legends = plt.getAllCurves(just_legend=True)
        if legend in legends:
            # modelIndex = cdict['modelIndex']
            # plt.getLegendsDockWidget()._legendWidget.model().removeRow(modelIndex.row())
            plt.removeCurve(legend)

    def common_legend_events_2d(self, cdict):
        for item in self.DATA_NAMES:
            for plots in [self.compare_plots, self.noise_plots]:
                if item in plots:
                    plt = plots[item][0]
                    plt.blockSignals(True)
                    if cdict['event'] == 'renameCurve':
                        self.rename_legend_2d(plt, oldLegend=cdict['legend'], newLegend=cdict['newLegend'])
                        self.reshow_legends(plt)
                    if cdict['event'] == 'removeCurve':
                        self.remove_legend_2d(plt, cdict['legend'])
                        self.reshow_legends(plt)
                    if cdict['event'] == 'offsetX':
                        plt.getLegendsDockWidget().offset_x(cdict['legend'], cdict['offset'],
                                                            absolute=cdict['absolute'])
                    plt.blockSignals(False)

    @staticmethod
    def rename_legend_2d(plt, oldLegend, newLegend):
        legends = plt.getAllImages(just_legend=True)
        if oldLegend in legends:
            plt.getLegendsDockWidget().renameImage(oldLegend, newLegend)

    @staticmethod
    def remove_legend_2d(plt, legend):
        legends = plt.getAllImages(just_legend=True)
        if legend in legends:
            plt.removeImage(legend)

    def rename_legend_stats(self, oldLegend, newLegend):
        for key, val in self.id_name_map.items():
            if val == oldLegend:
                self.id_name_map[key] = newLegend
                break
        # Update stats legend:
        for col, item in enumerate(self.tblHeader):
            if item == 'Id':
                for row in np.arange(self.tblStats.rowCount()):
                    if self.tblStats.item(row, col) is not None and self.tblStats.item(row, col).text() == oldLegend:
                        self.tblStats.item(row, col).setText(newLegend)


class EditTitle(QDialog):
    def __init__(self, parent=None, vals=[]):
        super().__init__(parent)

        self.setWindowTitle('Update curves')
        self.line_pos = QLineEdit(self)
        self.line_width = QLineEdit(self)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        if any(vals):
            if len(vals) > 0:
                self.line_pos.setText(vals[0])
            if len(vals) > 1:
                self.line_width.setText(vals[2])

        layout = QFormLayout(self)
        layout.addRow('Line positions ({}):'.format(vals[1]), self.line_pos)
        layout.addRow('Line width ({}):'.format(vals[3]), self.line_width)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.line_pos.text(), self.line_width.text())
