# coding=utf-8
'''

Use the CCD calibration method from pyFAI to retreive the pixel size from a calibration grid.
The distrotion maps are also produced.

https://pyfai.readthedocs.io/en/master/usage/tutorial/Detector/CCD_Calibration/CCD_calibration.html

@author: fraperri
'''

from Orange.widgets import gui, widget
from Orange.widgets.utils.signals import Input, Output
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QGridLayout, QScrollArea, QSizePolicy as Policy, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from orangewidget.settings import Setting
from orangewidget.widget import Msg

from pylost_widgets.widgets._PylostBase import PylostBase, PylostWidgets

CHECKED = True
try:
    import pyFAI
except ImportError:
    CHECKED = False

if CHECKED:
    try:  # depends if the version of pyFAI you are using
        from pyFAI.watershed import InverseWatershed
    except:
        from pyFAI.ext.watershed import InverseWatershed
        # Version of pyFAI newer than feb 2016
    try:
        from pyFAI.bilinear import Bilinear
    except:
        from pyFAI.ext.bilinear import Bilinear

from scipy.optimize import minimize
from scipy import ndimage, signal
from scipy.spatial import distance_matrix
from scipy.signal import find_peaks
from scipy.interpolate import griddata
import numpy as np
import time


class PlotCanvas(FigureCanvas):
    def __init__(self, parent, widget):
        self.fig = Figure(figsize=(6.32, 6.32), edgecolor='gray', linewidth=0.1, tight_layout=True)
        self.axes = self.fig.subplots(1, 2)
        super().__init__(self.fig)
        self.setParent(parent)
        self.widget = widget

    def _draw(self):
        self.draw()
        self.flush_events()
        # plt.pause(0.0001)

    def clear_plots(self):
        self.fig.clf()
        self.axes = self.fig.subplots(1, 2)
        self._draw()

    def share_axis(self):
        self.axes[1].sharex(self.axes[0])
        self.axes[1].sharey(self.axes[0])

    def draw_plots(self, plot_type='distortion'):
        if self.widget.img is None:
            return
        self.clear_plots()
        if 'cnv' in plot_type:
            self.share_axis()
            self.axes[0].imshow(self.widget.img, interpolation="nearest", origin="lower")
            self.axes[0].set_title("Measurement")
            self.axes[1].imshow(self.widget.cnv, interpolation="nearest", origin="lower")
            self.axes[1].set_title(
                f"Convolution (kernel size: {self.widget.kernel_size}x{self.widget.kernel_size} pixels)")
        if 'pop' in plot_type:
            self.axes[0].hist(np.log10(self.widget.population), 20)
            ylim = self.axes[0].get_ylim()
            self.axes[0].plot((self.widget.peaks_lower_limit,) * 2, ylim, ':r')
            self.axes[0].set_title("Peaks population")
            self.axes[0].annotate('lower limit',
                                  xy=(self.widget.peaks_lower_limit, ylim[1] * 0.90), xycoords='data',
                                  xytext=(25, 15), textcoords='offset points',
                                  arrowprops=dict(arrowstyle="->"),
                                  horizontalalignment='left', verticalalignment='bottom')
            xlim = self.axes[0].get_xlim()
            self.axes[0].axvspan(xlim[0], self.widget.peaks_lower_limit, ymax=ylim[1] / self.axes[0].get_ylim()[1],
                                 color='gray', alpha=0.50)
            self.axes[1].hist(self.widget.pairwise.ravel(), 200, range=(0, 200))
            self.axes[1].set_title("Pair-wise distribution function")
            num = len(self.widget.peaks)
            self.axes[1].annotate(f'{num} remaining peaks',
                                  xy=(0, len(self.widget.peaks) * 1.05), xycoords='data',
                                  xytext=(0, 30), textcoords='offset points',
                                  arrowprops=dict(arrowstyle="->"),
                                  horizontalalignment='left', verticalalignment='top')
            self.axes[1].annotate(f'first neighbours',
                                  xy=(self.widget.dot_separation[0], self.widget.first_neighbours * 1.05),
                                  xycoords='data',
                                  xytext=(11, 30), textcoords='offset points',
                                  arrowprops=dict(arrowstyle="->"),
                                  horizontalalignment='left', verticalalignment='top')
            ylim = self.axes[1].get_ylim()
            self.axes[1].plot((self.widget.dot_separation[1],) * 2, ylim, ':r')
            self.axes[1].annotate(f'max separation used: {int(self.widget.dot_separation[1])} pix',
                                  xy=(self.widget.dot_separation[1], ylim[1] * 0.90), xycoords='data',
                                  xytext=(21, 15), textcoords='offset points',
                                  arrowprops=dict(arrowstyle="->"),
                                  horizontalalignment='left', verticalalignment='bottom')
        if 'pos' in plot_type:
            self.share_axis()
            data, peaks_m, peaks_c = (self.widget.img, self.widget.measured, self.widget.expected)
            diff = -self.widget.delta
            if self.widget.use_cnv:
                data = self.widget.cnv
            self.axes[0].imshow(data, interpolation="nearest", origin="lower", alpha=0.65)
            self.axes[0].plot(peaks_c[:, 0], peaks_c[:, 1], "or", markersize=3)
            self.axes[0].plot(peaks_m[:, 0], peaks_m[:, 1], "oy", markersize=1)
            self.axes[0].set_title("Peak position: expected (red) and measured (yellow)")
            self.axes[1].imshow(data, interpolation="nearest", origin="lower", alpha=0.65)
            self.axes[1].quiver(peaks_c[:, 0], peaks_c[:, 1], diff[:, 0], diff[:, 1],
                                angles='xy', scale=self.widget.quiver_scale)
            self.axes[1].set_title("Quiver plot")
        if 'map' in plot_type:
            self.share_axis()
            im_ratio = self.widget.distortion[0].shape[0] / self.widget.distortion[0].shape[1]
            vmin = min([np.nanmin(self.widget.distortion[0]), np.nanmin(self.widget.distortion[1])])
            vmax = max([np.nanmax(self.widget.distortion[0]), np.nanmax(self.widget.distortion[1])])
            im0 = self.axes[0].imshow(self.widget.distortion[0], interpolation="nearest", origin="lower",
                                      vmin=vmin, vmax=vmax, cmap='Greys_r')
            self.axes[0].set_title(r"$\delta$x (um)")
            im_ratio = self.widget.distortion[0].shape[0] / self.widget.distortion[0].shape[1]
            self.fig.colorbar(im0, ax=self.axes[0], fraction=0.046 * im_ratio, pad=0.04)
            im1 = self.axes[1].imshow(self.widget.distortion[1], interpolation="nearest", origin="lower",
                                      vmin=vmin, vmax=vmax, cmap='Greys_r')
            self.axes[1].set_title(r"$\delta$y (um)")
            self.fig.colorbar(im1, ax=self.axes[1], fraction=0.046 * im_ratio, pad=0.04)
        self._draw()


class OWGrid(PylostWidgets, PylostBase):
    name = 'Calibration grid'
    description = 'PyFAI is not present.\nUse \"pip install pyFAI\"\nor \"conda install -c conda-forge pyfai\".'
    if CHECKED:
        description = 'Use PyFAI ESRF code to extract the distortion from a regular grid. Pixel size is also calulated.'
    icon = "../icons/grid.png"
    priority = 74

    class Inputs:
        data = Input('data', dict, auto_summary=False)

    class Outputs:
        dxy = Output('distortion map', list, default=True, auto_summary=False)
        ps = Output('pixel size', float, auto_summary=False)

    want_main_area = 0

    peaks_lower_limit = Setting(0.8, schema_only=True)
    use_cnv = Setting(True, schema_only=True)
    grid_res = Setting(1, schema_only=True)
    plot = Setting(2, schema_only=True)
    quiver_scale = Setting(5, schema_only=True)

    class Error(widget.OWWidget.Error):
        scans_not_found = Msg("No scan data is available.")
        unknown = Msg("Error:\n{}")

    def __init__(self):
        super().__init__()
        PylostBase.__init__(self)

        box = super().init_info(module=False)

        prm_layout = QGridLayout(self)
        prm_box = gui.widgetBox(self.controlArea, "Parameters", margin=10, orientation=prm_layout, addSpace=False)
        prm_widget = QWidget(prm_box)
        ch0 = gui.checkBox(prm_widget, self, 'use_cnv', 'Use convolution')
        ch1 = gui.doubleSpin(prm_widget, self, value='peaks_lower_limit', minv=0, maxv=10, step=0.01,
                             sizePolicy=Policy(Policy.Fixed, Policy.Fixed), )
        ch1_lbl = gui.label(prm_widget, self, 'peaks lower limit')
        ch2 = gui.doubleSpin(prm_widget, self, value='grid_res', minv=0, maxv=10, step=0.001,
                             sizePolicy=Policy(Policy.Fixed, Policy.Fixed), )
        ch2_lbl = gui.label(prm_widget, self, 'grid resolution (mm)')
        ch3 = gui.doubleSpin(prm_widget, self, value='quiver_scale', minv=0.1, maxv=100, step=0.1,
                             sizePolicy=Policy(Policy.Fixed, Policy.Fixed), callback=self.scale_changed)
        ch3_lbl = gui.label(prm_widget, self, 'Quiver scaling')
        self.calculate = gui.button(prm_widget, self, 'Distortion', callback=self.calibration,
                                    autoDefault=False, stretch=1, sizePolicy=(Policy.Fixed, Policy.Fixed))
        prm_layout.addWidget(ch0, 0, 0, Qt.AlignVCenter | Qt.AlignLeft)
        prm_layout.addWidget(ch1_lbl, 0, 1, Qt.AlignVCenter | Qt.AlignRight)
        prm_layout.addWidget(ch1, 0, 2, Qt.AlignVCenter | Qt.AlignLeft)
        prm_layout.addWidget(ch2_lbl, 0, 3, Qt.AlignVCenter | Qt.AlignRight)
        prm_layout.addWidget(ch2, 0, 4, Qt.AlignVCenter | Qt.AlignLeft)
        prm_layout.addWidget(ch3_lbl, 0, 5, Qt.AlignVCenter | Qt.AlignRight)
        prm_layout.addWidget(ch3, 0, 6, Qt.AlignVCenter | Qt.AlignLeft)
        prm_layout.addWidget(self.calculate, 0, 8, Qt.AlignVCenter | Qt.AlignRight)
        self.calculate.setEnabled(CHECKED)

        tool_layout = QGridLayout(self)
        tool_box = gui.widgetBox(self.controlArea, '', margin=0, orientation=tool_layout, addSpace=False)
        self.PlotBoxLayout = gui.hBox(self.controlArea, '', stretch=10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.PlotBoxLayout.layout().addWidget(scroll)
        self.canvas = PlotCanvas(scroll, self)
        scroll.setWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, None, coordinates=True)
        tool_layout.addWidget(self.toolbar)

        plot_type_layout = QGridLayout(self)
        plot_type_box = gui.widgetBox(self.controlArea, "Current plot", margin=10, orientation=plot_type_layout,
                                      addSpace=False)
        vbox = gui.radioButtons(None, self, "plot", callback=self.change_plot, box=True,
                                addSpace=False, addToLayout=False, orientation=Qt.Horizontal)
        rb1 = gui.appendRadioButton(vbox, "Convolution maps", addToLayout=False)
        rb2 = gui.appendRadioButton(vbox, "Population plots", addToLayout=False)
        rb3 = gui.appendRadioButton(vbox, "Peaks positions", addToLayout=False)
        rb4 = gui.appendRadioButton(vbox, "Distortion maps", addToLayout=False)
        plot_type_layout.addWidget(rb1, 0, 0, Qt.AlignVCenter | Qt.AlignHCenter)
        plot_type_layout.addWidget(rb2, 0, 1, Qt.AlignVCenter | Qt.AlignHCenter)
        plot_type_layout.addWidget(rb3, 0, 2, Qt.AlignVCenter | Qt.AlignHCenter)
        plot_type_layout.addWidget(rb4, 0, 3, Qt.AlignVCenter | Qt.AlignHCenter)

        self.img = None
        self.cnv = None
        self.population = None
        self.pairwise = None
        self.measured = None
        self.expected = None
        self.delta = None
        self.distortion = None
        self.pixel_res = None
        self.file_res = None
        self.peaks = None
        self.first_neighbours = None
        self.dot_separation = None
        self.kernel_size = None

    def sizeHint(self):
        return QSize(632, 1264)

    @Inputs.data
    def set_data(self, data):
        self.setStatusMessage('')
        self.clear_messages()
        super().set_data(data, update_names=True, show_all_default_names=True)
        if data is not None:
            self.data_in = data.get('height', None)
        else:
            self.Outputs.dxy.send(None)
            self.Outputs.ps.send(None)

    def load_data(self, multi=False):
        super().load_data()
        self.canvas.clear_plots()
        self.Outputs.dxy.send(None)
        self.Outputs.ps.send(None)

    def update_comment(self, comment, prefix=''):
        super().update_comment(comment, prefix='Pixel size')
        self.setStatusMessage(f'Pixel size: {self.pixel_res * 1e3:.3f} um (file: {self.file_res * 1e3:.3f} um)')

    def change_plot(self):
        plots = ('cnv', 'pop', 'pos', 'map')
        self.canvas.draw_plots(plots[self.plot])

    def scale_changed(self):
        if self.plot == 2:
            self.change_plot()

    def calibration(self):
        self.setStatusMessage('')
        self.clear_messages()
        if self.data_in is None or len(self.data_in) == 0:
            return
        self.info.set_output_summary('Calculating...')
        self.img = self.data_in.value
        self.file_res = self.data_in.pix_size[0].to('mm').value
        unit = self.data_in.pix_size[0].unit
        try:
            self.get_distortion()
            self.plot = 2
            self.change_plot()
            self.Outputs.dxy.send(self.distortion)
            # self.data_in.pix_size[0]
            self.Outputs.ps.send(self.pixel_res * unit)
        except Exception as e:
            print(e)
            self.info.set_output_summary('Error')
            self.Error.unknown(repr(e))
            self.plot = 1
            self.change_plot()

    def get_distortion(self):
        print("Working with pyFAI version: %s\n" % pyFAI.version)
        start_time = time.perf_counter()

        if self.use_cnv:
            print('\nUsing convolution.')
        else:
            print('\nNot using convolution.')

        # kernel
        # size = 11 #Odd of course
        size = int(0.66 / self.file_res)
        if (size % 2) == 0:
            size += 1
        print(f'\nSize of the kernel:{size}x{size} pixels')
        self.kernel_size = size

        center = (size - 1) // 2
        y, x = np.ogrid[-center:center + 1, -center:center + 1]
        r2 = x * x + y * y
        kernel = (r2 <= (center + 0.5) ** 2).astype(float)
        kernel /= kernel.sum()
        mini = (kernel > 0).sum()
        print("Number of points in the kernel: %s" % mini)

        # convolution
        self.cnv = signal.convolve2d(self.img, kernel, mode="same")

        # segments
        if self.use_cnv:
            iw = InverseWatershed(self.cnv)
        else:
            iw = InverseWatershed(self.img)
        iw.init()
        iw.merge_singleton()
        all_regions = set(iw.regions.values())
        regions = [i for i in all_regions if i.size > mini]

        # Look for the maximum value in each region to be able to segment accordingly
        self.population = [i.maxi for i in regions]

        print("Number of region segmented: %s" % len(all_regions))
        print("Number of large enough regions : %s" % len(regions))
        self.peaks = [(i.index // self.img.shape[-1], i.index % self.img.shape[-1]) for i in regions if
                      (i.maxi) > 10 ** self.peaks_lower_limit]
        print("Number of remaining peaks: %s" % len(self.peaks))
        peaks_raw = np.array(self.peaks)
        # refine peaks extraction
        if self.use_cnv:
            bl = Bilinear(self.cnv)
        else:
            bl = Bilinear(self.img)
        ref_peaks = [bl.local_maxi(p) for p in self.peaks]
        peaks_ref = np.array(ref_peaks)

        # pair-wise
        # Nota, pyFAI uses **C-coordinates** so they come out as (y,x) and not the usual (x,y).
        # This notation helps us to remind the order
        yx = np.array(ref_peaks)
        # pairwise distance calculation using scipy.spatial.distance_matrix
        dist = distance_matrix(peaks_ref, peaks_ref)
        self.pairwise = dist
        hist = np.histogram(dist.ravel(), 200, range=(0, 200))[0]
        sep, dic = find_peaks(hist, threshold=hist[0] / 10)
        dot_separation = sep[0] + (sep[1] - sep[0]) / 2
        print(f'\nMaximum separation between pair-wise dots: {dot_separation} pixels')

        # We define here a data-type for each peak (called center) with 4 neighbours (called north, east, south and west).
        point_type = np.dtype([('center_y', float), ('center_x', float),
                               ('east_y', float), ('east_x', float),
                               ('west_y', float), ('west_x', float),
                               ('north_y', float), ('north_x', float),
                               ('south_y', float), ('south_x', float)])

        neig = np.logical_and(dist > 5.0, dist < dot_separation)
        valid = (neig.sum(axis=-1) == 4).sum()
        self.dot_separation = (sep[0], dot_separation)
        neigh = dic.get('left_thresholds', [0, 0])
        self.first_neighbours = int(neigh[0])
        print("\nThere are %i control point with exactly 4 first neigbours" % valid)
        # This initializes an empty structure to be populated
        point = np.zeros(valid, point_type)
        # Populate the structure: we use a loop as it loops only over 400 points
        h = -1
        for i, center in enumerate(peaks_ref):
            if neig[i].sum() != 4:
                continue
            h += 1
            point[h]["center_y"], point[h]["center_x"] = center
            for j in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                tmp = []
                for k in np.where(neig[i]):
                    curr = yx[k]
                    tmp.append(np.dot(curr - center, j))
                    l = np.argmax(tmp)
                    y, x = peaks_ref[np.where(neig[i])][l]
                    if j == (0, 1):
                        point[h]["east_y"], point[h]["east_x"] = y, x
                    elif j == (0, -1):
                        point[h]["west_y"], point[h]["west_x"] = y, x
                    elif j == (1, 0):
                        point[h]["north_y"], point[h]["north_x"] = y, x
                    elif j == (-1, 0):
                        point[h]["south_y"], point[h]["south_x"] = y, x

        # Select the initial guess for the center:

        # Most intense peak:
        # m = max([i for i in regions], key=lambda i:i.maxi)
        # Cx, Cy = m.index%self.img.shape[-1],m.index//self.img.shape[-1]
        # Cx, Cy = point["center_x"].mean(), point["center_y"].mean() #Centroid of all points
        # beam center
        Cx, Cy = tuple(i // 2 for i in self.img.shape)  # detector center
        print("\nThe guessed center is at (%s, %s)" % (Cx, Cy))
        # Get the nearest point from centroid:
        d2 = ((point["center_x"] - Cx) ** 2 + (point["center_y"] - Cy) ** 2)
        best = d2.argmin()
        Op = point[best]
        Ox, Oy = Op["center_x"], Op["center_y"]
        print("The center is at (%s, %s)" % (Ox, Oy))
        # Calculate the average vector along the 4 main axes
        Xx = (point[:]["east_x"] - point[:]["center_x"]).mean()
        Xy = (point[:]["east_y"] - point[:]["center_y"]).mean()
        Yx = (point[:]["north_x"] - point[:]["center_x"]).mean()
        Yy = (point[:]["north_y"] - point[:]["center_y"]).mean()
        # print("\nThe X vector is is at (%s, %s)"%(Xx, Xy))
        # print("The Y vector is is at (%s, %s)"%(Yx, Yy))
        # print("X has an angle of %s deg"%np.rad2deg(np.arctan2(Xy, Xx)))
        # print("Y has an angle of %s deg"%np.rad2deg(np.arctan2(Yy, Yx)))
        # print("The XY angle is %s deg"%np.rad2deg(np.arctan2(Yy, Yx)-np.arctan2(Xy, Xx)))
        x = point[:]["center_x"] - Ox
        y = point[:]["center_y"] - Oy
        xy = np.vstack((x, y))
        R = np.array([[Xx, Yx], [Xy, Yy]])
        iR = np.linalg.inv(R)
        IJ = np.dot(iR, xy).T
        Xmin = IJ[:, 0].min()
        Xmax = IJ[:, 0].max()
        Ymin = IJ[:, 1].min()
        Ymax = IJ[:, 1].max()
        # print("\nXmin/max",Xmin, Xmax)
        # print("Ymin/max",Ymin,Ymax)
        pitch = self.grid_res * 1e-3  # mm distance between holes
        print(f"\nMaximum error versus integer: %s * pitch size (%.3f mm)" % (abs(IJ - IJ.round()).max(), pitch * 1e3))
        # pixel size
        Py = pitch * np.sqrt((Yx ** 2 - Xx ** 2) / ((Xy * Yx) ** 2 - (Xx * Yy) ** 2))
        Px = np.sqrt((pitch ** 2 - (Xy * Py) ** 2) / Xx ** 2)
        print("\nPixel size in average: x:%.3f microns, y: %.3f microns" % (Px * 1e6, Py * 1e6))

        # optimization
        # Measured peaks (all!), needs to flip x<->y
        peaks_m = np.empty_like(peaks_ref)
        peaks_m[:, 1] = peaks_ref[:, 0]
        peaks_m[:, 0] = peaks_ref[:, 1]
        self.measured = peaks_m
        # parameter set for optimization:
        P0 = [Ox, Oy, Xx, Yx, Xy, Yy]
        P = np.array(P0)

        def to_hole(P, pixels):
            "Translate pixel -> hole"
            T = np.atleast_2d(P[:2])
            R = P[2:].reshape((2, 2))
            # Transformation matrix from pixel to holes:
            hole = np.dot(np.linalg.inv(R), (pixels - T).T).T
            return hole

        def to_pix(P, holes):
            "Translate hole -> pixel"
            T = np.atleast_2d(P[:2])
            R = P[2:].reshape((2, 2))
            # Transformation from index points (holes) to pixel coordinates:
            pix = np.dot(R, holes.T).T + T
            return pix

        def error(P):
            "Error function"
            hole_float = to_hole(P, peaks_m)
            hole_int = hole_float.round()
            delta = hole_float - hole_int
            delta2 = (delta ** 2).sum()
            return delta2

        # print("\nTotal inital error ", error(P), P0)
        holes = to_hole(P, peaks_m)
        print("Maximum initial error versus integer: %s * pitch size (%.3f mm)" % (
            abs(holes - holes.round()).max(), pitch * 1e3))
        res = minimize(error, P)
        # print("total Final error ", error(res.x),res.x)
        holes = to_hole(res.x, peaks_m)
        print("Maximum final error versus integer: %s * pitch size (%.3f mm)" % (
            abs(holes - holes.round()).max(), pitch * 1e3))

        peaks_c = to_pix(res.x, to_hole(res.x, peaks_m).round())
        self.expected = peaks_c

        Ox, Oy, Xx, Yx, Xy, Yy = res.x
        Py = pitch * np.sqrt((Yx ** 2 - Xx ** 2) / ((Xy * Yx) ** 2 - (Xx * Yy) ** 2))
        Px = np.sqrt((pitch ** 2 - (Xy * Py) ** 2) / Xx ** 2)
        self.pixel_res = 1000 * (Px + Py) / 2
        msg1 = "\nOptimized pixel size in average: x: %.3f um, y: %.3f um, average: %.3f um" % (
            Px * 1e6, Py * 1e6, self.pixel_res * 1e3)
        print(msg1)

        diff = (self.file_res - self.pixel_res) / self.file_res
        msg2 = f'Lateral resolution given by the file: {self.file_res * 1e3:.3f} um'
        print(msg2)
        print(f'\ndifference: {diff * 1e2:.2f}%')

        self.info.set_output_summary(msg1 + ', ' + msg2)
        self.update_comment(msg1 + '\n' + msg2)

        # interpolation
        grid_x, grid_y = np.mgrid[0:self.img.shape[0] + 1, 0:self.img.shape[1] + 1]
        self.delta = peaks_c - peaks_m
        diffmax = np.max(np.sqrt(np.power(self.delta[:, 0], 2) + np.power(self.delta[:, 1], 2)))
        self.quiver_scale = self.dot_separation[1] / 2 * diffmax

        # we use peaks_res instead of peaks_m to be in y,x coordinates, not x,y
        delta_x = griddata(peaks_ref, self.delta[:, 0], (grid_x, grid_y), method='cubic')
        delta_y = griddata(peaks_ref, self.delta[:, 1], (grid_x, grid_y), method='cubic')

        # From http://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays
        def fill(data, invalid=None):
            """
            Replace the value of invalid 'data' cells (indicated by 'invalid')
            by the value of the nearest valid data cell

            Input:
                data:    numpy array of any dimension
                invalid: a binary array of same shape as 'data'. True cells set where data
                         value should be replaced.
                         If None (default), use: invalid  = np.isnan(data)

            Output:
                Return a filled array.
            """
            if invalid is None:
                invalid = np.isnan(data)
            ind = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
            return data[tuple(ind)]

        # self.distortion = (fill(delta_x), fill(delta_y))
        self.distortion = (delta_x, delta_y)

        print(f"\n\nExecution time: {time.perf_counter() - start_time:.3f} s")
