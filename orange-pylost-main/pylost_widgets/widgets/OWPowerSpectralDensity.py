# coding=utf-8
import numpy as np
from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QSizePolicy as Policy, QTabWidget, QToolBar, QLabel, QFileDialog
from astropy.units import Quantity
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.util.MetrologyData import MetrologyData
from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets

from PyQt5.QtWidgets import QGridLayout, QScrollArea
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as Toolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

from scipy.fft import fft, fft2, fftfreq
from scipy.signal import periodogram
import astropy.units as u

# from matplotlib.backends.qt_compat import (
#     QtCore, QtGui, QtWidgets, _enum, _to_int,
#     _setDevicePixelRatio, _devicePixelRatioF,
#     )

# from pathlib import Path
# import os

# class CustomToolbar(Toolbar, QToolBar):
#     # .\matplotlib\backends\backend_qt.py
#     # .\matplotlib\backend_bases.py
#     def __init__(self, canvas, parent, coordinates=True):
#     # basic additional tool implementation with icon in matplotlib folder
#         # toolitems = [*self.toolitems]
#         # toolitems.insert(
#         #     # Add 'customize' action after 'subplots'
#         #     [name for name, *_ in toolitems].index("Test") + 1,
#         #     ('Test', 'test tool','test', 'print_test'))
#
#         self._mask = None
#         self._profiles = None
#         super().__init__(canvas, parent, False)
#
#         while parent.parent() is not None:
#             parent = parent.parent()
#         self.main_window = parent
#
#     # basic additional tool implementation with icon in local folder
#         custom_tools = [
#             (None, None, None, None),
#             ('Export', 'Export PSD & CPSD to csv file', 'export', 'export_data'),
#             ]
#         self.register_tools(custom_tools)
#
#         custom_tools = [
#             (None, None, None, None),
#             ('Window', 'Export window', 'export_w', 'export_window'),
#         ]
#         self.register_tools(custom_tools)
#
#     # coordinates after tools
#         self.coordinates = coordinates
#         if self.coordinates:
#             self.locLabel = QLabel("", self)
#             self.locLabel.setAlignment(Qt.AlignmentFlag(
#                 _to_int(_enum("QtCore.Qt.AlignmentFlag").AlignRight) |
#                 _to_int(_enum("QtCore.Qt.AlignmentFlag").AlignVCenter)))
#             self.locLabel.setSizePolicy(QtWidgets.QSizePolicy(
#                 _enum("QtWidgets.QSizePolicy.Policy").Expanding,
#                 _enum("QtWidgets.QSizePolicy.Policy").Ignored,
#             ))
#             labelAction = self.addWidget(self.locLabel)
#             labelAction.setVisible(True)
#
#     def _customicon(self, name): # from backend_qt
#         """
#         Construct a `.QIcon` from an image file *name*, including the extension
#         and relative to Matplotlib's "images" data directory.
#         """
#         name = name.replace('.png', '_large.png')
#         p = str(Path(__file__).with_name("icons")) # local path
#         pm = QtGui.QPixmap(str(Path(p, name)))
#         _setDevicePixelRatio(pm, _devicePixelRatioF(self))
#         if self.palette().color(self.backgroundRole()).value() < 128:
#             icon_color = self.palette().color(self.foregroundRole())
#             mask = pm.createMaskFromColor(
#                 QtGui.QColor('black'),
#                 _enum("QtCore.Qt.MaskMode").MaskOutColor)
#             pm.fill(icon_color)
#             pm.setMask(mask)
#         return QtGui.QIcon(pm)
#
#     def register_tools(self, tools): # from backend_qt
#         for text, tooltip_text, image_file, callback in tools:
#             if text is None:
#                 self.addSeparator()
#             else: # custom icon in local folder
#                 a = self.addAction(self._customicon(image_file + '.png'),
#                                    text, getattr(self, callback))
#                 self._actions[callback] = a
#                 if callback in ['zoom', 'pan']:
#                     a.setCheckable(True)
#                 if tooltip_text is not None:
#                     a.setToolTip(tooltip_text)
#
#     def export_data(self):
#         try:
#             start_file = os.path.expanduser("~/")
#             if len(self.canvas.ow.data_in) > 0:
#                 start_file = self.canvas.ow.data_in.get('comment_log', start_file)
#                 pathidx = start_file.find('Data loaded from file ')
#                 if pathidx != -1:
#                     start_file = start_file.split('Data loaded from file ')[-1][:-1]
#                     start_file = str(Path(start_file).parent)
#                     if 'sequence' in start_file:
#                         start_file = start_file[10:]
#             file, selfilter = QFileDialog.getSaveFileName(None, directory=start_file, filter='csv file (*.csv *.txt);;all files(*.*)')
#             if len(file) > 0:
#                 plots = ('height', 'slopes_x', 'slopes_y')
#                 plot_type = plots[self.canvas.ow.plot]
#                 calc = self.canvas.ow.data_nD.get(plot_type, None)
#                 units = self.canvas.ow.data_nD.get('units', '')
#                 unit = 1e6 * u.urad if 'slope' in plot_type else Quantity(1, unit=u.m).to(units[0])
#                 if calc is None:
#                     return
#                 psd = calc['psd']
#                 if psd is not None:
#                     psd = psd * unit * unit * unit
#                 csp = calc['csp']
#                 if csp is not None:
#                     csp = csp * unit
#                 freq = calc.get('freq', None)
#                 if psd is None or freq is None:
#                     return
#                 freq = Quantity(freq, 1/u.m).to(1/units[1])
#                 header = 'Spatial Frequency (1/mm), ' + f'Power Spectral Density ({psd.unit}), ' + f'Cumulative Spectral Power ({csp.unit})'
#                 np.savetxt(file, np.asfarray((freq.value, psd.value, csp.value),).T, header=header, delimiter=',')
#         except:
#             print('could not export plots to csv file.')
#
#     def export_window(self):
#         try:
#             start_file = os.path.expanduser("~/")
#             if len(self.canvas.ow.data_in) > 0:
#                 start_file = self.canvas.ow.data_in.get('comment_log', start_file)
#                 pathidx = start_file.find('Data loaded from file ')
#                 if pathidx != -1:
#                     start_file = start_file.split('Data loaded from file ')[-1][:-1]
#                     start_file = str(Path(start_file).parent)
#                     if 'sequence' in start_file:
#                         start_file = start_file[10:]
#             file, selfilter = QFileDialog.getSaveFileName(None, directory=start_file, filter='csv file (*.csv *.txt);;all files(*.*)')
#             if len(file) > 0:
#                 plots = ('height', 'slopes_x', 'slopes_y')
#                 plot_type = plots[self.canvas.ow.plot]
#                 calc = self.canvas.ow.data_nD.get(plot_type, None)
#                 units = self.canvas.ow.data_nD.get('units', '')
#                 unit = 1e6 * u.urad if 'slope' in plot_type else Quantity(1, unit=u.m).to(units[0])
#                 if calc is None:
#                     return
#                 N, T, window = self.canvas.ow.windows
#                 if window is None:
#                     return
#                 window = window.ravel()
#                 x = np.arange(N) * T
#                 x = x - np.mean(x)
#                 x = Quantity(x, u.m).to(units[1])
#                 w_params = self.canvas.ow.window()
#                 prms = ''
#                 if isinstance(w_params, str):
#                     wname = w_params
#                 else:
#                     wname, prms = w_params
#                 header = f'mirror coordinates ({x.unit}), ' + wname + f' {prms} (normalized)'
#                 np.savetxt(file, np.asfarray((x.value, window),).T, header=header, delimiter=',')
#         except Exception as e:
#             print('could not export plots to csv file.', e)


class PlotCanvas(FigureCanvas):
    def __init__(self, parent, ow):
        self.fig = Figure(figsize=(6.32, 6.32), edgecolor='gray', linewidth=0.1, tight_layout=True)
        self.axes = self.fig.subplots(1, 2)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ow = ow

    def _draw(self):
        self.draw()
        self.flush_events()
        # plt.pause(0.0001)

    def clear_plots(self):
        self.fig.clf()
        self.axes = self.fig.subplots(1, 2)
        self._draw()

    def draw_plots(self, plot_type='height'):
        self.clear_plots()
        if len(self.ow.data_nD) == 0:
            return
        try:
            units = self.ow.data_nD.get('units', None)
            unit = 1e6 * u.urad if 'slope' in plot_type else Quantity(1, unit=u.m).to(units[0])
            calc = self.ow.data_nD.get(plot_type, None)
            if calc is None:
                return
            psd = calc['psd']
            if psd is not None:
                psd = psd * unit * unit * unit
            csp = calc['csp']
            if csp is not None:
                csp = csp * unit
            dim = self.ow.data_nD.get('dim', None)
            freq = calc.get('freq', None)
            if psd is None or freq is None:
                return
            freq = Quantity(freq, 1/u.m).to(1/units[1])
            psd_legend = f'Power Spectral Density ({psd.unit})'
            if len(dim) > 1:
                self.fig.clf()
                self.axes = self.fig.subplots(1, 1)
                im0 = self.axes.imshow(psd.value, interpolation="nearest", origin="lower", norm=LogNorm())
                self.axes.set_axis_off()
                # im_ratio = psd.shape[0] / psd.shape[1]
                # self.fig.colorbar(im0, ax=self.axes, fraction=0.046 * im_ratio, pad=0.04)
            elif len(dim) == 1:
                self.axes[0].loglog(freq[1:], psd.value[1:])
                self.axes[0].set_xlabel(f'Spatial Frequency (1/mm)')
                # self.axes[0].set_ylabel(psd_legend)
                self.axes[0].set_title(psd_legend, fontdict={'fontsize': 12})
                self.axes[0].grid(color='silver', linestyle='-', linewidth=1, which="both")
                if csp is not None:
                    self.axes[1].semilogx(freq[1:], csp.value[1:])
                    self.axes[1].yaxis.tick_right()
                    csp_legend = f'Cumulative Spectral Power ({csp.unit})'
                    self.axes[1].set_xlabel(f'Spatial Frequency (1/mm)')
                    # self.axes[1].set_ylabel(csp_legend)
                    self.axes[1].set_title(csp_legend, fontdict={'fontsize': 12})
                    self.axes[1].yaxis.set_label_position("right")
                    self.axes[1].grid(color='silver', linestyle='-', linewidth=1, which="both")
            self._draw()
        except ValueError:
            pass


class OWPowerSpectralDensity(PylostWidgets, PylostBase):
    name = 'Power Spectral Density'
    description = 'Calculate power spectral density in 1D (along last axis).'
    icon = "../icons/psd.svg"
    priority = 33

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        data = Output('data', dict, auto_summary=False)

    AVG_1D, SUPERFLAT, SUPERFLAT2, AVG_2D, ALL_2D = range(5)
    NONE, BLACKMAN, HANNING, HAMMING, BLACKMAN_HARRIS, KAISER, TUKEY = range(7)
    WINDOWS = ['None', 'Blackman', 'Hanning', 'Hamming', 'Blackman-Harris', 'Kaiser', 'Tukey']

    want_main_area = 0
    module = Setting('', schema_only=True)
    scan_name = Setting('')
    win = Setting(1, schema_only=True)
    option = Setting(AVG_1D, schema_only=True)
    kaiser_beta = Setting(10, schema_only=True)
    tukey_alpha = Setting(0.2, schema_only=True)
    flip_csp = Setting(False, schema_only=True)

    fmin = Setting(0.0, schema_only=True)
    fmax = Setting(0.0, schema_only=True)

    plot = Setting(1, schema_only=True)

    tabs = QTabWidget()  # TODO: update_tabs can't be set to False
    dataViewers = {}

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=True, module_callback=self.change_module, scans=True,
                                scans_callback=self.change_scan)
        self.btnApply = gui.button(box, self, 'Calculate PSD', callback=self.apply, autoDefault=False, stretch=1,
                                   sizePolicy=(Policy.Fixed, Policy.Fixed))

        hbox = gui.hBox(self.controlArea, 'Options', stretch=2)
        box = gui.vBox(hbox)
        options = ['PSD 1D curves average', 'Superflat script (hgt->slp in fourier space)', 'Superflat script (PSD on slopes)', 'PSD 2D linewise average', 'Areal PSD 2D']
        self.combo = gui.comboBox(box, self, "option", label='Type of PSD:', callback=self.apply, orientation=Qt.Horizontal,
                                  sizePolicy=(Policy.Fixed, Policy.Fixed), items=options)
        gui.comboBox(box, self, "win", label='Window:', callback=[self.change_win, self.apply],
                     orientation=Qt.Horizontal, sizePolicy=(Policy.Fixed, Policy.Fixed), items=self.WINDOWS)
        gui.doubleSpin(box, self, value='kaiser_beta', label='Kaiser beta:', minv=0, maxv=1000, step=0.01,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.doubleSpin(box, self, value='tukey_alpha', label='Tukey alpha:', minv=0, maxv=1, step=0.01,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.checkBox(box, self, 'flip_csp', 'Flip CSP direction (integrate right to left)', callback=self.calc_csp,
                     sizePolicy=(Policy.Fixed, Policy.Fixed))

        box = gui.vBox(hbox)
        self.label = gui.label(box, self, '')
        gui.label(box, self, 'N - Number of points along x in data, d - pixel size in data')

        box = gui.vBox(self.controlArea, 'CSPD cutoff frequencies', stretch=2)
        hbox = gui.hBox(box, sizePolicy=(Policy.Fixed, Policy.Fixed))
        gui.doubleSpin(hbox, self, value='fmin', label='low limit:', minv=0, maxv=10000, step=0.0001,
                       callback=self.calc_csp, sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.freq_lbl1 = gui.label(hbox, self, '1/length')
        gui.doubleSpin(hbox, self, value='fmax', label='high limit:', minv=0, maxv=10000, step=0.0001,
                       callback=self.calc_csp, sizePolicy=(Policy.Fixed, Policy.Fixed))
        self.freq_lbl2 = gui.label(hbox, self, '1/length')
        # gui.button(hbox, self, 'Update', sizePolicy=(Policy.Fixed, Policy.Fixed))

        # # Data viewer
        plot_type_layout = QGridLayout(self)
        gui.widgetBox(self.controlArea, "Current plot", margin=10, orientation=plot_type_layout, addSpace=False)
        vbox = gui.radioButtons(None, self, "plot", callback=self.change_plot, box=True,
                                addSpace=False, addToLayout=False, orientation=Qt.Horizontal)
        rb1 = gui.appendRadioButton(vbox, "Heights", addToLayout=False)
        rb2 = gui.appendRadioButton(vbox, "Slopes X", addToLayout=False)
        rb3 = gui.appendRadioButton(vbox, "Slopes Y", addToLayout=False)
        plot_type_layout.addWidget(rb1, 0, 0, Qt.AlignVCenter | Qt.AlignHCenter)
        plot_type_layout.addWidget(rb2, 0, 1, Qt.AlignVCenter | Qt.AlignHCenter)
        plot_type_layout.addWidget(rb3, 0, 2, Qt.AlignVCenter | Qt.AlignHCenter)

        tool_layout = QGridLayout(self)
        gui.widgetBox(self.controlArea, 'Data viewer', margin=0, orientation=tool_layout, addSpace=False)
        self.PlotBoxLayout = gui.hBox(self.controlArea, '', stretch=10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.PlotBoxLayout.layout().addWidget(scroll)
        self.canvas = PlotCanvas(scroll, self)
        scroll.setWidget(self.canvas)
        self.toolbar = Toolbar(self.canvas, self.canvas, coordinates=True)
        tool_layout.addWidget(self.toolbar)

        self.change_win()

        self.data_nD = {}

        self.windows = []

    def sizeHint(self):
        return QSize(500, 700)

    @Inputs.data
    def set_data(self, data):
        super().set_data(data, update_tabs=True, show_all_default_names=True)
        if data is None:
            self.Outputs.data.send(None)

    def load_data(self, multi=False):
        super().load_data()
        self.change_module()

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Power Spectral Density')

    def change_win(self):
        if self.win == self.KAISER:
            self.controls.kaiser_beta.parent().show()
        else:
            self.controls.kaiser_beta.parent().hide()
        if self.win == self.TUKEY:
            self.controls.tukey_alpha.parent().show()
        else:
            self.controls.tukey_alpha.parent().hide()

    def calc_csp(self, item=None):
        if self.option == self.ALL_2D:
            return
        redraw = False
        if item is None:
            redraw = True
            item = ('height', 'slopes_x', 'slopes_y')[self.plot]
        freq = self.data_nD[item]['freq']
        q = np.isfinite(freq)
        if self.fmin != 0:
            q = np.multiply(q, freq >= self.fmin*1e3)
        if self.fmax != 0:
            q = np.multiply(q, freq <= self.fmax*1e3)
        yq = np.where(q, self.data_nD[item]['psd'], 0)
        dim = 1
        for d in self.data_nD['dim']:
            dim = d * dim
        if self.flip_csp:
            yq = np.flip(yq)
        yq = np.sqrt(np.cumsum(yq) / dim)
        if self.flip_csp:
            yq = np.flip(yq)
        self.data_nD[item]['csp'] = yq
        if redraw:
            self.change_plot()

    def change_plot(self):
        plots = ('height', 'slopes_x', 'slopes_y')
        self.canvas.draw_plots(plots[self.plot])

    def apply(self):
        try:
            nd = False
            for item in self.DATA_NAMES:
                if item in self.data_in:
                    nd = self.data_in[item].ndim > 1
                    if nd:
                        break
            self.combo.model().item(1).setEnabled(nd)
            self.combo.model().item(2).setEnabled(nd)

            self.label.setText('')
            super().apply_scans(autoclose=False)
            self.Outputs.data.send(self.data_out)
            self.change_plot()
        except Exception as e:
            self.Error.unknown(str(e))

    def apply_scan(self, scan, scan_name=None, comment=''):
        scan_full = self.full_dataset(scan)
        return super().apply_scan(scan_full, scan_name, comment)

    def apply_scan_item(self, Z, comment='', item=None):
        dims = super().get_detector_dimensions(Z)
        axes = dims.nonzero()[0]
        val = Z.si.value
        hgt_unit = None
        if 'height' in item:
            hgt_unit = Z.unit, Z.get_axis_val_items_detector()[-1].unit
        if Z.ndim > 1 and self.option >= self.AVG_2D:
            if not any(axes):
                axes = [-2, -1]
            N = [Z.shape[axes[-2]], Z.shape[axes[-1]]]
            T = self.pix_size[-2:]
            unit_f = ['', '']
            arr = Z.get_axis_val_items_detector()[-2:]
            for i in range(2):
                if np.any(arr[i].value):
                    T[i] = np.nanmean(np.diff(arr[i])).si.value
                    unit_f[i] = '1/mm'
                    self.freq_lbl1.setText(unit_f[0])
                    self.freq_lbl2.setText(unit_f[0])

            w1 = self.get_window(Z.ndim, N[-1], axes[-1])
            w2 = self.get_window(Z.ndim, N[-2], axes[-2])

            # self.windows = [(N[-1], T[-1], w1), (N[-2], T[-2], w2)]
            self.windows = (N[-1], T[-1], w1)

            zf = fft2(val * w1 * w2, axes=axes[-2:])
            slc = [slice(None)] * Z.ndim
            slc[axes[-1]] = slice(0, N[-1] // 2)
            slc[axes[-2]] = slice(0, N[-2] // 2)
            Zret = (2.0 * T[-2] / N[-2]) * (2.0 * T[-1] / N[-1]) * np.abs(zf[slc]) ** 2
            slc[axes[-1]] = 0
            slc[axes[-2]] = 0
            Zret[slc] = Zret[slc] / 2
            xf = [fftfreq(N[-2], T[-2])[:N[-2] // 2], fftfreq(N[-1], T[-1])[:N[-1] // 2]]

            comment = 'PSD 2D along last two axis'

            if self.option == self.ALL_2D:
                tot_z2 = np.nansum(Z ** 2) / (N[-1] * N[-2])
                tot_psd = np.nansum(Zret[:, 1:]) / (N[-1] * T[-1] * Z.unit * N[-2] * T[-2])
                tot_psd *= 1e6 * u.urad * 1e6 * u.urad if 'slope' in item else 1e9 * u.nm * 1e9 * u.nm
                txt = '{}{} : Sum_psd/Nd = {:.4f}, Sum_z2/N={:.4f}\n'.format(self.label.text(), item, tot_psd, tot_z2)
                txt = txt + f'  Cumulative Spectral Power: {np.sqrt(tot_psd):.4f}\n'
                self.label.setText(txt)
                self.data_nD['dim'] = [N[i] * T[i] for i, _ in enumerate(N)]
                if hgt_unit is not None:
                    self.data_nD['units'] = hgt_unit
                self.data_nD[item] = {'psd': Zret, 'csp': None, 'freq':xf[-1]}

            elif self.option == self.AVG_2D:
                axis = -1
                other = -2
                # if 'slopes_y' in item:
                #     axis = -2
                #     other = -1
                Zret = np.nanmean(Zret, axis=axes[other]) / (T[other])  # Alcock et al.  is it correct?
                comment = 'PSD 2D along last two axis and average linewise'
                tot_z2 = np.nansum(Z ** 2) / N[axis]
                tot_psd = np.nansum(Zret[1:]) / (N[axis] * T[axis])
                tot_psd *= 1e6 * u.urad * 1e6 * u.urad if 'slope' in item else 1e9 * u.nm * 1e9 * u.nm
                txt = '{}{} : Sum_psd/Nd = {:.4f}, Sum_z2/N={:.4f}\n'.format(self.label.text(), item, tot_psd, tot_z2)
                txt = txt + f'  Cumulative Spectral Power: {np.sqrt(tot_psd):.4f}\n'
                self.label.setText(txt)
                self.data_nD['dim'] = (N[axis] * T[axis],)
                if hgt_unit is not None:
                    self.data_nD['units'] = hgt_unit
                self.data_nD[item] = {'psd': Zret[1:], 'csp': None, 'freq':xf[axis][1:]}
                self.calc_csp(item)

        else:
            if self.option == self.SUPERFLAT:  #  https://gitlab.synchrotron-soleil.fr/OPTIQUE/leaps/superflat_scripts
                comment = 'scipy.signal.periodogram on height' + '; Window : {}'.format(self.WINDOWS[self.win])
                if 'slope' not in item:
                    N = Z.shape[axes[-1]]
                    arr = Z.get_axis_val_items_detector()[-1]
                    unit_f = '1/mm'
                    self.freq_lbl1.setText(unit_f)
                    self.freq_lbl2.setText(unit_f)
                    T = np.nanmean(np.diff(arr)).si.value  # pixel size
                    xf = 1.0 / T  # sampling rate in x

                    (sfx, h2xpsd) = periodogram(val, xf, window=self.window(), return_onesided=True)  # one sided psd with tukey (0.2) taper
                    h1psd = np.mean(h2xpsd, axis=0)  # average over l
                    s1xpsd = h1psd * (2 * np.pi * sfx) ** 2  # height to slope in fourier space
                    # (sfy, h2ypsd) = periodogram(val.T, xf, window=self.window(), return_onesided=True)  # one sided psd with tukey (0.2) taper
                    # s1ypsd = np.mean(h2ypsd, axis=0) * (2 * np.pi * sfy) ** 2  # height to slope in fourier space

                    comment = f'scipy.signal.periodogram on {item}'
                    tot_z2 = (np.nansum(Z ** 2) / N)
                    unit = 1e9 * u.nm
                    tot_psd = (np.nansum(h1psd) / (N * T)) * unit * unit
                    txt = '{}{} : Sum_psd/Nd = {:.4f}, Sum_z2/N={:.4f}\n'.format(self.label.text(), 'height', tot_psd, tot_z2)
                    txt = txt + f'  Cumulative Spectral Power: {np.sqrt(tot_psd):.4f}\n'
                    self.label.setText(txt)
                    unit = 1e6 * u.urad
                    tot_psd = (np.nansum(s1xpsd) / (N * T)) * unit * unit
                    txt = '{}{} : Sum_psd/Nd = {:.4f}, Sum_z2/N={:.4f}\n'.format(self.label.text(), 'slopes_x', tot_psd, tot_z2)
                    txt = txt + f'  Cumulative Spectral Power: {np.sqrt(tot_psd):.4f}\n'
                    self.label.setText(txt)
                    # tot_psd = (np.nansum(s1ypsd) / (N * T)) * unit * unit
                    # txt = '{}{} : Sum_psd/Nd = {:.4f}, Sum_z2/N={:.4f}\n'.format(self.label.text(), 'slopes_x', tot_psd, tot_z2)
                    # txt = txt + f'  Cumulative Spectral Power: {np.sqrt(tot_psd):.4f}\n'
                    # self.label.setText(txt)
                    self.data_nD['dim'] = (N * T,)
                    if hgt_unit is not None:
                        self.data_nD['units'] = hgt_unit
                    self.data_nD['height'] = {'psd': h1psd, 'csp': None, 'freq': sfx}
                    self.calc_csp('height')
                    self.data_nD['slopes_x'] = {'psd': s1xpsd, 'csp': None, 'freq': sfx}
                    self.calc_csp('slopes_x')
                    self.data_nD['slopes_y'] = {'psd': None, 'csp': None, 'freq': None}
                    # self.data_nD['slopes_y'] = {'psd': s1ypsd, 'csp': None, 'freq': sfy}
                    # self.calc_csp('slopes_y')
                    comment += '; Window : {}'.format(self.WINDOWS[self.win])

            elif self.option == self.SUPERFLAT2:  #  PSD directly on slopes in real space instead of fourier space
                N = Z.shape[axes[-1]]
                arr = Z.get_axis_val_items_detector()[-1]
                unit_f = '1/mm'
                self.freq_lbl1.setText(unit_f)
                self.freq_lbl2.setText(unit_f)
                T = np.nanmean(np.diff(arr)).si.value  # pixel size
                xf = 1.0 / T  # sampling rate in x

                # if 'slopes_y' in item:
                #     val = val.T
                (sf, h2psd) = periodogram(val, xf, window=self.window(), return_onesided=True)  # one sided psd with tukey (0.2) taper
                Zret = np.mean(h2psd, axis=0)  # average over l

                comment = f'scipy.signal.periodogram on {item}'
                tot_z2 = (np.nansum(Z ** 2) / N)
                tot_psd = (np.nansum(Zret) / (N * T))
                tot_psd *= 1e6 * u.urad * 1e6 * u.urad if 'slope' in item else 1e9 * u.nm * 1e9 * u.nm
                txt = '{}{} : Sum_psd/Nd = {:.4f}, Sum_z2/N={:.4f}\n'.format(self.label.text(), item, tot_psd, tot_z2)
                txt = txt + f'  Cumulative Spectral Power: {np.sqrt(tot_psd):.4f}\n'
                self.label.setText(txt)

                self.data_nD['dim'] = (N * T,)
                if hgt_unit is not None:
                    self.data_nD['units'] = hgt_unit
                self.data_nD[item] = {'psd': Zret, 'csp': None, 'freq': sf}
                self.calc_csp(item)
                comment += '; Window : {}'.format(self.WINDOWS[self.win])

            else:
                if not any(axes):
                    axes = [-1]
                axis = -1
                other = -2
                # if 'slopes_y' in item:
                #     axis = -2
                #     other = -1
                N = Z.shape[axes[axis]]
                T = self.pix_size[axis]
                unit_f = ''
                arr = Z.get_axis_val_items_detector()[axis]
                if np.any(arr.value):
                    T = np.nanmean(np.diff(arr)).si.value  # pixel size
                    unit_f = '1/mm'
                    self.freq_lbl1.setText(unit_f)
                    self.freq_lbl2.setText(unit_f)

                w = self.get_window(Z.ndim, N, axes[axis])
                self.windows = (N, T, w)
                zf = fft(val * w, axis=axes[axis])
                slc = [slice(None)] * Z.ndim
                slc[axes[axis]] = slice(0, N // 2)
                Zret = (2.0 * T / N) * np.abs(zf[slc]) ** 2
                slc[axes[axis]] = 0
                Zret[slc] = Zret[slc] / 2
                xf = fftfreq(N, T)[:N // 2]

                comment = 'PSD 1D along last axis'

                tot_z2 = np.nansum(Z ** 2) / N
                if self.option == self.AVG_1D and Zret.ndim > 1:
                    Zret = np.nanmean(Zret, axis=axes[other])
                    comment = 'PSD 1D along last axis and average PSD curves'
                tot_psd = np.nansum(Zret[1:]) / (N * T)
                tot_psd *= 1e6 * u.urad * 1e6 * u.urad if 'slope' in item else 1e9 * u.nm * 1e9 * u.nm
                txt = '{}{} : Sum_psd/Nd = {:.4f}, Sum_z2/N={:.4f}\n'.format(self.label.text(), item, tot_psd, tot_z2)
                txt = txt + f'  Cumulative Spectral Power: {np.sqrt(tot_psd):.4f}\n'
                self.label.setText(txt)

                self.data_nD['dim'] = (N * T,)
                if hgt_unit is not None:
                    self.data_nD['units'] = hgt_unit
                self.data_nD[item] = {'psd': Zret[1:], 'csp': None, 'freq': xf[1:]}
                self.calc_csp(item)

        comment += '; Window : {}'.format(self.WINDOWS[self.win])

        obj = Quantity(self.data_nD['height']['psd'], unit=u.m*u.m*u.m)
        if len(obj.shape) > 1:
            return Z, comment
        zunit = self.data_nD['units'][0]*self.data_nD['units'][0]*self.data_nD['units'][0]
        obj = obj.to(zunit)
        freq = Quantity(self.data_nD['height']['freq'], unit=1/u.m)
        freq = freq.to(1/self.data_nD['units'][1])
        obj = obj.view(MetrologyData)
        # obj._set_unit(self.data_nD['units'][0])
        obj._set_dim_detector([True])  # 1D
        obj._set_pix_size(np.nanmean(np.diff(freq)))
        obj._set_index_list(np.linspace(0, obj.shape[0], num=obj.shape[0], endpoint=False))
        obj._set_axis_names(['X'])
        obj._set_axis_values([''])
        obj._set_init_shape(obj.shape[0])
        obj._set_motors([])
        obj._set_flags(Z._flags)
        obj.add_flag('1D_height_psd_data', True)

        return obj, comment

    def get_window(self, ndim, N, axis):
        from scipy.signal.windows import blackman, hann, hamming, blackmanharris, kaiser, tukey
        w = 1
        if self.win != self.NONE:
            shp = np.ones((ndim,), dtype=int)
            shp[axis] = N
            if self.win == self.BLACKMAN:
                w = blackman(N).reshape(shp)
            elif self.win == self.HANNING:
                w = hann(N).reshape(shp)
            elif self.win == self.HAMMING:
                w = hamming(N).reshape(shp)
            elif self.win == self.BLACKMAN_HARRIS:
                w = blackmanharris(N).reshape(shp)
            elif self.win == self.KAISER:
                w = kaiser(N, beta=self.kaiser_beta).reshape(shp)
            elif self.win == self.TUKEY:
                w = tukey(N, alpha=self.tukey_alpha).reshape(shp)
        return w

    def window(self):
        if self.win != self.NONE:
            if self.win == self.BLACKMAN:
                return 'blackman'
            elif self.win == self.HANNING:
                return 'hann'
            elif self.win == self.HAMMING:
                return 'hamming'
            elif self.win == self.BLACKMAN_HARRIS:
                return 'blackmanharris'
            elif self.win == self.KAISER:
                return ('kaiser', self.kaiser_beta)
            elif self.win == self.TUKEY:
                return ('tukey', self.tukey_alpha)
