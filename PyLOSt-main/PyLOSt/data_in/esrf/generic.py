# -*- coding: utf-8 -*-
"""

http://hplgit.github.io/primer.html/doc/pub/class/._class-solarized003.html

https://docs.python.org/3/glossary.html#term-bytecode

https://realpython.com/python-descriptors/

https://stackoverflow.com/questions/2278426/inner-classes-how-can-i-get-the-outer-class-object-at-construction-time

https://medium.com/@vadimpushtaev/decorator-inside-python-class-1e74d23107f6

https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#spline-interpolation
"""

# pylint: disable=C0103, C0115, C0116
# pylint: disable=C0415
# pylint: disable=R0902, R0903, R0904, R0913, R0914

import warnings

from time import perf_counter_ns

import copy

import numpy as np

from numpy.polynomial.polynomial import polyfit, polyval
from scipy.interpolate import interp1d, BSpline
from scipy.integrate import simps, romb
from scipy.ndimage import rotate

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap as lsc

try:
    import swift.units as u
except ImportError:
    from . import units as u


class ProtoData:
    def __init__(self, coords: (list, tuple), values: (list, tuple, np.array), units:dict, source=None):
        if isinstance(coords, tuple):
            coords = list(coords)
            for i, c in enumerate(coords):
                coords[i] = np.float64(c)
        self.coords = coords.copy()
        # self.values = np.float64(values).copy()
        # self.coords = np.asfarray(coords, dtype='float64')
        self.values = np.array(values, dtype='float64')
        # if isinstance(units, tuple):
        #     units = {'coords': units[0], 'values': [1]}
        # if units is None:
        #     units = {'coords': None, 'values': None}
        self.units = copy.deepcopy(units)
        self.source = source

        self._raw_shape = None
        self.mask = None

        initial = None
        if 'initial' in self.__dict__:
            initial = self.__dict__['initial']
        self.initial = initial

        self.ellipse = None
        self.p = None
        self.q = None
        self.theta = None
        self.fit = None
        self.terms = None

    # ----properties----
    @property
    def kind(self):
        unit_str = str(self.values_unit)
        if 'rad' in unit_str.lower():
            return 'slopes'
        if 'm-1' in unit_str.lower():
            return 'curvature'
        return 'heights'

    @property
    def x(self):
        if self.values.ndim > 1:
            return self.coords[0]
        return self.coords

    @property
    def x_unit(self):
        return self.coords_unit

    @property
    def z(self):
        return self.values

    @property
    def z_unit(self):
        return self.values_unit

    @property
    def shape(self):
        if self.values is None:
            return self._raw_shape
        return self.values.shape

    @property
    def u(self):
        return self.units

    @property
    def coords_unit(self):
        return self.units['coords']

    @property
    def default_coords_unit(self):
        return self.initial.units['coords']

    @property
    def values_unit(self):
        return self.units['values']

    @property
    def default_values_unit(self):
        return self.initial.units['values']

    @property
    def hasnan(self):
        return np.all(np.isnan(self.values))

    # ----common functions----
    def apply(self):
        self.initial.coords = self.coords.copy()
        self.initial.values = self.values.copy()
        self.initial.units = self.units.copy()
        return self

    def duplicate(self):
        # # cls = type(self)
        cls = Profile
        if self.values.ndim > 1:
            cls = Surface
        return cls(self.coords, self.values, self.units, self.source)

    def reload(self):
        if self.initial is not None:
            if self.values.ndim > 1:
                for i, coords in enumerate(self.initial.coords):
                    self.coords[i] = np.copy(coords)
            else:
                self.coords = np.copy(self.initial.coords)
            self.values = np.copy(self.initial.values)
            self.units = self.initial.units.copy()

    def automasking(self, threshold=None, apply=False):
        if threshold is None:
            threshold = 3.0 * self.rms
        self.mask = np.fabs(self.values - self.median) < threshold  # XXX: or mean_removal ?
        newval = np.where(self.mask, self.values, np.nan)
        if apply:
            self.values = newval
        return newval

    def level_data(self):
        """Level the central value (along the X-axis) to zero and return the new values.
        Perform a small interpolation if case of even number of points.
        """
        # center = 0.0
        size = len(self.values)
        coords = self.coords
        values = self.values
        if self.values.ndim == 2:
            # size = size[0]
            coords = coords[0]
            values = np.nanmean(values, axis=1)
        nb_points = max(4, int(size * 0.01))
        if size % 2 > 0:  # odd --> no problem
            center = values[int(size / 2)]
        else:  # even --> quadratic interpolation over the central nb_points
            window = int(nb_points / 2)
            first = int(size / 2) - window
            last = int(size / 2) + window
            interp = interp1d(coords[first:last],
                              values[first:last],
                              kind='quadratic')
            center = interp(np.nanmean(coords))
        self.values -= center
        return self.values

    # ----units functions----
    def auto_units(self):
        self.auto_coords_unit()
        self.auto_values_unit()

    def auto_coords_unit(self):
        pv = np.nanmax(self.coords) - np.nanmin(self.coords)
        _, new_unit = self.coords_unit.auto(pv / 2)
        self.change_coords_unit(new_unit)

    def auto_values_unit(self):
        _, new_unit = self.values_unit.auto(self.pv / 2)
        self.change_values_unit(new_unit)

    def reset_units(self):
        self.reset_coords_unit()
        self.reset_values_unit()

    def reset_coords_unit(self):
        self.change_coords_unit(self.initial.coords_unit)

    def reset_values_unit(self):
        self.change_values_unit(self.initial.values_unit)

    def change_defaults_units(self, new_units):
        if isinstance(new_units, dict):
            self.initial.units = new_units
            return
        self.initial.units['coords'] = new_units[0]
        self.initial.units['values'] = new_units[1]

    def change_units(self, new_units):
        if isinstance(new_units, dict):
            self.change_coords_unit(new_units['coords'])
            self.change_values_unit(new_units['values'])
            return
        self.change_coords_unit(new_units[0])
        self.change_values_unit(new_units[1])

    def change_coords_unit(self, new_unit):
        if new_unit is self.coords_unit:
            return
        self.coords = new_unit(self.coords, self.coords_unit)
        self.units['coords'] = new_unit

    def change_values_unit(self, new_unit):
        if new_unit is self.values_unit:
            return
        self.values = new_unit(self.values, self.values_unit)
        self.units['values'] = new_unit

    def units_to_SI(self):
        self.coords_unit_to_SI()
        self.values_unit_to_SI()

    def coords_unit_to_SI(self):
        self.change_coords_unit(self.coords_unit.SI_unit)

    def values_unit_to_SI(self):
        self.change_values_unit(self.values_unit.SI_unit)

    # ----ellipse----
    def set_ellipse(self, p, q, theta, reload=True):
        """p and q in meter, theta in mrad"""
        try:
            from esrf.data.ellipse import Ellipse
        except ImportError:
            from .ellipse import Ellipse
        if reload:
            initial = self.initial
            if initial is None:
                initial = self
            self.ellipse = Ellipse(initial, p, q, theta)
        else:
            self.ellipse = Ellipse(self, p, q, theta)
        return self.ellipse

    def ellipse_removal(self, p, q, theta, optimize=True, optimization=('q', 'theta')):
        """p and q in meter, theta in mrad"""
        if not self.ellipse:
            self.set_ellipse(p, q, theta)
        print('ellipse removal')
        print(self.ellipse.measurement.values_unit)
        print(self.ellipse.measurement.pv)
        if optimize:
            self.ellipse.optimize(optimization)
        self.change_values_unit(self.ellipse.residues.values_unit)
        self.values = self.ellipse.residues.values
        self.p = self.ellipse.p
        self.q = self.ellipse.q
        self.theta = self.ellipse.theta
        self.fit = 'ellipse'
        self.terms = 'Ellipse'

    # ----analysis----
    def gravity_correction(self, length, thickness, distance, material='Silicon', operation='subtract'):
        """Length, thickness, distance in mm."""
        try:
            from esrf.data import gravity
            grav = gravity.model(self, length, thickness, distance, material)
            if 'add' in operation.lower():
                grav.values = -grav.values
            delta = grav.mean - self.mean
            self.values = self.values + grav.values - delta
            return grav
        except ImportError:
            return None

    def center_coordinates(self):
        if self.values.ndim > 1:
            for i, coords in enumerate(self.coords):
                self.coords[i] = coords - np.nanmean(coords)
        else:
            self.coords = self.coords - np.nanmean(self.coords)
        return self.coords, self.values

    def flip(self, axis='x'):
        ax = 0
        if 'y' in axis.lower():
            ax = 1
        if self.values.ndim > 1:
            self.values = np.flip(self.values, ax)
        else:
            self.values = np.flip(self.values)
        if 'slope' in self.kind.lower():
            self.values = np.negative(self.values)
        return self.coords, self.values

    def min_removal(self):
        """Substract the minimum of the values."""
        self.values -= self.min
        return self.values

    def mean_removal(self):
        """Substract the average of the values."""
        self.values -= self.mean
        return self.values

    @property
    def max(self):
        return np.nanmax(self.values)

    @property
    def min(self):
        return np.nanmin(self.values)

    @property
    def mean(self):
        return np.nanmean(self.values)

    @property
    def pv(self):
        return self.max - self.min

    @property
    def rms(self):
        return np.nanstd(self.values)

    @property
    def median(self):
        return np.nanmedian(self.values)


class Profile(ProtoData):
    def __init__(self, coords: (list, tuple), values: (list, tuple, np.array), units:dict, source=None):
        super().__init__(coords, values, units, source)

    def __call__(self):
        return self.coords, self.values

    # ----maths----
    def derivative(self, method='gradient'):
        """methods: gradient, cubicbspline, """
        res = self.duplicate()
        res.units_to_SI()
        if method == 'gradient':
            res.values = np.gradient(res.values, res.coords)
        elif method == 'cubicbspline':
            spline = BSpline(*res(), k=3)
            # deriv = spline.derivative()
            res.values = spline.derivative()(res.coords)
            # res.coords = res.coords[:-1] + res.mean_step/2
        else:
            return None
        if 'height' in self.kind.lower():
            res.units['values'] = res.values_unit.angle
        elif 'slope' in self.kind.lower():
            res.units['values'] = res.values_unit.curvature
        res.auto_units()
        res.initial = res.duplicate()
        return res

    def integral(self, method='simpson'):
        """methods: trapezoidal, simpson, romb, cubicbspline, """
        res = self.duplicate()
        res.units_to_SI()
        res.level_data()
        step = abs(res.mean_step)
        coords = res.coords  # - step/2
        coords = np.concatenate(([coords[0] - step], coords, [coords[-1] + step]))
        idx = np.arange(1, len(coords) - 1)
        if method == 'cubicbspline':
            spline = BSpline(*res(), k=3)
            integr = [spline.integrate(coords[i - 1], coords[i]) for i in idx]
        else:
            spline = BSpline(*res(), k=2)
            interpolate = spline(coords)
            if method == 'trapezoidal':
                integr = [np.trapz(interpolate[i - 1:i + 1], coords[i - 1:i + 1]) for i in idx]
            elif method == 'romberg':
                integr = [romb(interpolate[i - 1:i + 1], step) for i in idx]
            else:
                integr = [simps(interpolate[i - 1:i + 1], coords[i - 1:i + 1]) for i in idx]
        res.values = np.cumsum(integr)
        if 'slope' in self.kind.lower():
            res.units['values'] = res.values_unit.length
        res.auto_units()
        res.initial = res.duplicate()
        return res

    def interpolation(self, new_coords, method='cubicbspline'):
        """method: cubicbspline"""
        res = Profile(self.coords, self.values, self.units, self.source)
        if method == 'cubicbspline':
            spline = BSpline(*res(), k=3)
            res.values = spline(new_coords)
            res.coords = new_coords
        else:
            return None
        return res

    @staticmethod
    def theoretical(x, shape, p=None, q=None, theta=None, roc=None):
        """kind: 'elliptical', 'parabolic', 'spherical'    (SI units)"""
        x = x - np.nanmean(x)
        if 'spherical' not in shape.lower():
            try:
                from esrf.data.ellipse import Ellipse
            except ImportError:
                from .ellipse import Ellipse
            if 'parabolic' in shape.lower():
                p = 1e5
            slp = Ellipse.from_parameters(x, 'slope', p, q, theta)
            hgt = Ellipse.from_parameters(x, 'height', p, q, theta)
        elif 'spherical' in shape.lower():
            slp = x / roc
            slp = slp - np.nanmean(slp)
            hgt = 2 * x ** 2 / roc
            hgt = hgt - np.nanmin(hgt)
        else:
            return None
        return slp, hgt

    # ----analysis----
    @property
    def mean_step(self):
        return np.diff(self.coords).mean()

    def reverse(self):
        """slopes: Flip the X coordinates and negate the values."""
        self.coords = self.coords[::-1]
        self.values = self.values[::-1]
        if 'slope' in self.kind.lower():
            self.values = np.negative(self.values)
        return self.coords, self.values

    def offsetting(self, offset=0.0):
        # left = self.values[0] + np.diff(self.values[:3]).mean() * np.sign(self.coords[0])
        # right = self.values[-1] + np.diff(self.values[-3:]).mean() * np.sign(self.coords[-1])
        self.values = np.interp(self.coords + offset, self.coords, self.values,
                                # left=left, right=right,
                                )
        return self.values

    def subarray(self, xrangearray=None):
        """profile: Cut the data to the x coordinates given by [Xmin, Xmax]."""
        if xrangearray:
            xrangearray -= self.mean_step / 4.0
            split_at = self.coords.searchsorted(xrangearray)
            self.coords = self.coords[split_at[0]:split_at[1] + 1]
            self.values = self.values[split_at[0]:split_at[1] + 1]
        return self.coords, self.values

    def polynomial_removal(self, order=1):
        """Remove a polynomial fit to the values.
        Return the polynomial fit
        """
        if np.all(np.isnan(self.values)):
            return self.values
        mask = ~np.isnan(self.values)
        domain = np.linspace(0, 1, num=len(self.values))
        poly = polyfit(domain[mask], self.values[mask], order)
        self.values -= polyval(domain, poly)
        return polyval(domain, poly)

    @property
    def radius(self):
        # domain = np.linspace(0, 1, num=len(self.values))
        unit = u.rad
        if 'height' in self.kind.lower():
            unit = u.m
        pol = polyfit(u.m(self.coords, self.coords_unit),
                      unit(self.values, self.values_unit),
                      2)
        return 1 / (2 * pol[2])

    def plot(self, **kwargs):
        """Matplotlib 1D plot with some additional functions.

            Parameters :
                matplotlib.pyplot.plot arguments
                https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1)  # , figsize=(10,5))
        # fig.set_tight_layout(True)
        ax.plot(*self(), linewidth=1, **kwargs)
        plt.grid()
        plt.show()


class Surface(ProtoData):
    def __init__(self, coords: (list, tuple), values: (list, tuple, np.array), units:dict, source=None):
        super().__init__(coords, values, units, source)
        self._pixel_res = np.asfarray([np.nanmean(np.diff(coords[0])), np.nanmean(np.diff(coords[1]))])
        self._pixel_res = u.m(self._pixel_res, units['coords'])
        self.piston = None
        self.tilt_x = None
        self.tilt_y = None
        self.rot_xy = None
        self.majcyl = None
        self.mincyl = None
        self.radius = None

        self.fit = None
        self.terms = 'No terms'

        self.cmap = None
        self.cmap_prms = None

    def __call__(self):
        return self.values

    # ----overriding units functions----
    def auto_coords_unit(self):
        pv = (np.nanmax(self.x) - np.nanmin(self.x),
              np.nanmax(self.y) - np.nanmin(self.y))
        _, new_unit = self.x_unit.auto(max(pv))
        self.change_coords_unit(new_unit)

    def change_coords_unit(self, new_unit):
        if new_unit is self.coords_unit:
            return
        self.coords[0] = new_unit(self.coords[0], self.coords_unit)
        self.coords[1] = new_unit(self.coords[1], self.coords_unit)
        self.units['coords'] = new_unit

    # ----properties----
    @property
    def pixel_size(self):
        return self.units['coords'](self._pixel_res, u.m)

    @property
    def y(self):
        return self.coords[1]

    @property
    def y_unit(self):
        return self.units['coords']

    @property
    def meshgrid(self):
        return np.meshgrid(self.x, self.y, indexing='ij')

    @property
    def valid_mask(self):
        return ~np.isnan(self.values)

    @property
    def valid_size(self):
        return np.sum(self.valid_mask)

    @property
    def invalid_pts(self):
        return np.sum(~self.valid_mask)

    @property
    def invalid_pct(self):
        return 100 * self.invalid_pts / self.values.size

    # ----manipulation----
    def change_resolution(self, *new_pixel_size):
        """size in meter"""
        size_x, size_y = new_pixel_size
        shape_x, shape_y = self.z.shape
        self.coords[0] = np.linspace(0, shape_x * size_x, num=shape_x,
                                     endpoint=False, dtype=np.float64)
        self.coords[1] = np.linspace(0, shape_y * size_y, num=shape_y,
                                     endpoint=False, dtype=np.float64)
        self._pixel_res = new_pixel_size

    def rotate(self, angle=None, center_coordinates=True):
        """Rotation angle in degrees."""
        if angle is None:
            return
        self.values = rotate(self.values, angle=angle,
                             reshape=True, cval=np.nan,
                             axes=(1, 0), prefilter=False)
        size = self.values.shape
        self.coords[0] = np.linspace(0, self.pixel_size[0] * size[0], size[0],
                                     endpoint=False, dtype=np.float64)
        self.coords[0] = self.coords[0] * np.cos(angle * np.pi / 180)
        self.coords[1] = np.linspace(0, self.pixel_size[1] * size[1], size[1],
                                     endpoint=False, dtype=np.float64)
        if center_coordinates:
            self.center_coordinates()
        # self.dropnan()

    def rotation_auto(self, center_coordinates=True):
        print('finding best rotation angle minimizing cylinder...')
        from scipy.optimize import fminbound
        data = self.duplicate()
        data.initial = data.duplicate()

        def _rotate(deg):
            data.reload()
            data.rotate(deg, center_coordinates)
            data.cylinder_removal(reload=False, auto_units=False, verbose=False)
            _, res = data.derivative()
            print(deg, res.rms * 1e6)
            return res.rms * 1e6
            # print(deg, data.pv)
            # return data.pv

        toc = perf_counter_ns()
        popt = [np.nan,]
        for i in range(0, 1):
            # popt = data.cylinder_removal(reload=False, auto_units=False, verbose=False)
            popt = fminbound(_rotate, -1.0, 1.0, xtol=1e-4,
                             full_output=True, maxfun=100, disp=0)
        tic = perf_counter_ns()
        print(f'best rotation angle based on conic fit: {popt[0]:.3f}deg ({(tic - toc) * 1e-6:.0f}ms)')
        self.rotate(popt[0], center_coordinates)

    def dropnan(self):
        mask = ~np.all(np.isnan(self.values), axis=0)
        self.coords[1] = self.coords[1][mask]
        self.values = self.values[:, mask]
        mask = ~np.all(np.isnan(self.values), axis=1)
        self.coords[0] = self.coords[0][mask]
        self.values = self.values[mask, :]

    def coords2pix(self, center=None, edge=None, size=None, from_center=True, short=False, unit=None):
        if size is None:
            size = self.values.shape
        if center is None:
            position = edge
        else:
            position = center - np.asfarray(size) / 2
        if position is None:
            position = (0, 0)
        split_at = np.asfarray((position, size)).T
        if unit is not None:
            split_at = self.coords_unit(split_at, unit)
        res = self.pixel_size
        for i, c in enumerate(split_at):
            p, s = c - (res[i] / 4.0) * np.sign(c + 1e-12)
            if from_center:
                p = p + np.nanmean(self.coords[i])
            s = p + s
            split_at[i] = self.coords[i].searchsorted(p), self.coords[i].searchsorted(s)
        if short:
            split_at = split_at - [[0, 1], [0, 1]]
        return split_at.astype(int)

    # ----analysis----
    def subarray_pixel(self, xrangearray=None, yrangearray=None, apply=True):
        if xrangearray is None:
            xrangearray = (0, self.values.shape[0])
        if yrangearray is None:
            yrangearray = (0, self.values.shape[1])
        coords_x = self.coords[0][xrangearray[0]:xrangearray[1] + 1]
        coords_y = self.coords[1][yrangearray[0]:yrangearray[1] + 1]
        values = self.values[xrangearray[0]:xrangearray[1] + 1, yrangearray[0]:yrangearray[1] + 1]
        if apply:
            self.coords[0] = coords_x
            self.coords[1] = coords_y
            self.values = values
        return (coords_x, coords_y), values

    def subarray(self, center=None, edge=None, size=None, from_center=True, short=False, unit=None, apply=True):
        """Mask the data according to the position and the size.
        if center is None, use lower-left egde.
        """
        split_at = self.coords2pix(center, edge, size, from_center, short, unit)
        coords, values = self.subarray_pixel(*split_at, apply=apply)
        return coords, values, split_at

    def extract_profile(self, position=0, width=1, length=None, from_center=True, axis='x', unit=u.mm):
        ax = 0
        if 'y' in axis.lower():
            ax = 1
        if length is None:
            length = max(self.coords[ax]) - min(self.coords[ax])
        center = [0, position]
        size = [length, width]
        if ax == 1:
            center = center[::-1]
            size = size[::-1]
        if not from_center:
            center[~ax] -= np.nanmean(self.coords[~ax])
        coords, values, _ = self.subarray(center=center, size=size, from_center=True, short=False, unit=unit, apply=False)
        return Profile(coords[ax], np.nanmean(values, axis=~ax), unit)

    # ----maths----
    def derivative(self, origin='lower-left'):
        """methods: gradient
        origin: combination of 'lower' 'upper' 'right' 'left'  (default: 'lower-left')
        """
        res_x = self.duplicate()
        res_x.units_to_SI()
        res_y = self.duplicate()
        res_y.units_to_SI()
        y = res_y.y
        for i, line in enumerate(res_x.values.T):
            x = res_x.x
            if 'right' in origin.lower():
                x = x[::-1]
            res_x.values.T[i] = np.gradient(line, x)
            y = res_y.y
            if 'upper' in origin.lower():
                y = y[::-1]
        for i, line in enumerate(res_y.values):
            res_y.values[i] = np.gradient(line, y)
        if 'height' in self.kind.lower():
            res_x.units['values'] = res_x.values_unit.angle
            res_y.units['values'] = res_y.values_unit.angle
        elif 'slope' in self.kind.lower():
            res_x.units['values'] = res_x.values_unit.curvature
            res_y.units['values'] = res_y.values_unit.curvature
        res_x.auto_units()
        res_x.initial = res_x.duplicate()
        res_y.auto_units()
        res_y.initial = res_y.duplicate()
        return res_x, res_y

    @staticmethod
    def _plane2D(coef, x, y, z):
        A, B, C = coef
        return z - (A * x + B * y + C)

    @staticmethod
    def _conic2D(coef, x, y, xx, xy, yy, z):
        A, B, C, D, E, F = coef
        return z - (A * xx + B * xy + C * yy + D * x + E * y + F)

    @staticmethod
    def _sphere2D(coef, x, y, xx, yy, z):
        A, B, C, D = coef
        return z - (A * (xx + yy) + B * x + C * y + D)

    @staticmethod
    def _fit2D(x, y, z, fit='plane', verbose=True):
        tic = perf_counter_ns()
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        mask = np.isnan(z)
        one = np.ones((len(x),), dtype=np.float64)
        M = np.array([x, y, one]).T
        if fit == 'conic':
            M = np.c_[x * x, x * y, y * y, M]
        elif fit == 'sphere':
            M = np.c_[(x * x + y * y), M]
        if M is not None:
            # return np.linalg.lstsq(M[~mask], z[~mask], rcond=None)
            res = np.linalg.lstsq(M[~mask], z[~mask], rcond=None)
            toc = perf_counter_ns()
            if verbose:
                print(f'applying {fit} least square fit to data ({(toc - tic) * 1e-6:.0f}ms)')
            return res
        return None

    def plane_removal(self, reload=True, auto_units=True, verbose=True):
        if reload:
            self.reload()
        self.units_to_SI()
        x, y = self.meshgrid
        pla = Surface._fit2D(x, y, self.values, fit='plane', verbose=verbose)
        self.values = Surface._plane2D(pla[0], x, y, self.values)
        self.tilt_x, self.tilt_y, self.piston = pla[0]
        self.piston = self.units['height'](self.piston, u.m)
        self.tilt_x = self.units['angle'](self.tilt_x, u.rad)
        self.tilt_y = self.units['angle'](self.tilt_y, u.rad)
        self.rot_xy = None
        self.majcyl = None
        self.mincyl = None
        self.radius = None
        self.fit = 'plane'
        self.terms = 'Tilt'
        if auto_units:
            self.auto_units()
        # return piston, tilt_x, tilt_y

    def cylinder_removal(self, reload=True, auto_units=True, verbose=True):
        if reload:
            self.reload()
        self.units_to_SI()
        x, y = self.meshgrid
        cyl = Surface._fit2D(x, y, self.values, fit='conic', verbose=verbose)
        self.values = Surface._conic2D(cyl[0], x, y, x * x, x * y, y * y, self.values)
        A, B, C, D, E, F = cyl[0]
        theta = np.arctan(B / (A - C)) / 2
        if np.fabs(A) > np.fabs(C):
            if theta > 0:
                theta -= np.pi / 2
            elif theta < 0:
                theta += np.pi / 2
        self.rot_xy = self.units['angle'](theta, u.mrad)
        Arot = (A * np.cos(theta) * np.cos(theta)
                + B * np.cos(theta) * np.sin(theta)
                + C * np.sin(theta) * np.sin(theta))
        self.majcyl = self.units['radius'](1 / (2 * Arot), u.m)
        theta += np.pi / 2
        Arot = (A * np.cos(theta) * np.cos(theta)
                + B * np.cos(theta) * np.sin(theta)
                + C * np.sin(theta) * np.sin(theta))
        self.mincyl = self.units['radius'](1 / (2 * Arot), u.m)
        self.tilt_x = self.units['angle'](D, u.rad)
        self.tilt_y = self.units['angle'](E, u.rad)
        self.piston = self.units['height'](F, u.m)
        self.radius = None
        self.fit = 'conic'
        self.terms = 'Cylinder & Tilt'
        if auto_units:
            self.auto_units()
        # return piston, tilt_x, tilt_y, rot_xy, majcyl, mincyl

    def sphere_removal(self, reload=True, auto_units=True, verbose=True):
        if reload:
            self.reload()
        self.units_to_SI()
        x, y = self.meshgrid
        sph = Surface._fit2D(x, y, self.values, fit='sphere', verbose=verbose)
        A, B, C, D = sph[0]
        self.radius = self.units['radius'](1 / (2 * A), u.m)
        self.tilt_x = self.units['angle'](B, u.rad)
        self.tilt_y = self.units['angle'](C, u.rad)
        self.piston = self.units['height'](D, u.m)
        self.values = Surface._sphere2D(sph[0], x, y, x * x, y * y, self.values)
        self.rot_xy = None
        self.majcyl = None
        self.mincyl = None
        self.fit = 'sphere'
        self.terms = 'Sphere & Tilt'
        if auto_units:
            self.auto_units()
        # return piston, tilt_x, tilt_y, curv

    # ----export----
    @staticmethod
    def writeMetroprofile(data, pixel_size, path, frame_size=None, phase_res=1):
        """
        data heights array and pixel_size in meters,
        frame_size as tuple (width, height), data.shape by default
        phase_res: present in the Metropro file format v3, default 1 -> 32768
        """
        import datetime as dt
        try:
            from esrf.data.zygo import MetroProData
        except ImportError:
            from .zygo import MetroProData
        new_data = MetroProData()
        new_data.header['header_size'] = 4096
        # new_data.header['swinfo.date'] = dt.datetime.now().strftime('%a %b %d %H:%M:%S %Y')
        new_data.header['time_stamp'] = np.int32(dt.datetime.now().timestamp())
        new_data.header['comment'] = 'file created from imported data'
        new_data.header['source'] = 1
        new_data.header['lateral_res'] = pixel_size
        new_data._raw_shape = data.shape
        width, height = new_data._raw_shape
        new_data.header['cn_width'] = width
        new_data.header['cn_height'] = height
        if frame_size is not None:
            if width > frame_size[0] or height > frame_size[1]:
                print('Frame size not consistent with the data size, input ignored!')
            else:
                width, height = frame_size
                width = np.int32(width)  # BUG: pylost gives int16 --> overflow can happens
                height = np.int32(height)  # BUG: pylost gives int16 --> overflow can happens
        new_data.header['ac_width'] = width
        new_data.header['ac_height'] = height
        new_data.header['cn_n_bytes'] = width * height * 4  # BUG: pylost gives int16 --> overflow can happens
        new_data.header['camera_width'] = width
        new_data.header['camera_height'] = height
        new_data.header['wavelength_in'] = np.float64(6.327999813038332e-07)  # must be in meter
        new_data.header['intf_scale_factor'] = 0.5
        new_data.header['obliquity_factor'] = 1.0
        new_data.header['phase_res'] = phase_res
        new_data.header_dict_to_class()
        x = np.linspace(0, new_data._raw_shape[0] * new_data.header['lateral_res'],
                        num=new_data._raw_shape[0], endpoint=False, dtype=np.float64),
        y = np.linspace(0, new_data._raw_shape[1] * new_data.header['lateral_res'],
                        num=new_data._raw_shape[1], endpoint=False, dtype=np.float64)
        new_data.initial = Surface([x, y], data, new_data.units)
        super(ESRFOpticsLabData, new_data).__init__(  # pylint: disable=super-with-arguments
            new_data.initial.coords, new_data.initial.values,
            new_data.initial.units, new_data.source)
        new_data.writefile(path)
        print(f'  data saved in MetroPro format ({path}).')
        return new_data

    # ----roughness----
    @property
    def ra(self):
        return np.nansum(np.fabs(self.z)) / self.valid_size

    @property
    def rq(self):
        return self.rms

    @property
    def rp(self):
        return self.max

    @property
    def rv(self):
        return self.min

    @property
    def rt(self):
        return self.pv

    @property
    def rsk(self):
        return np.nansum(np.power(self.z, 3)) / (self.valid_size * np.power(self.rq, 3))

    @property
    def rku(self):
        return np.nansum(np.power(self.z, 4)) / (self.valid_size * np.power(self.rq, 4))

    @property
    def rz(self):
        return self.filt_rz(self.z)

    @staticmethod
    def filt_rz(surfacedata, num_pts=10, window=None):
        data = surfacedata.copy()
        if window is None:
            window = [-5, 5]
        shape = data.shape
        if len(shape) == 2:
            data = data.ravel()
        if len(shape) > 2:
            print('rz calc: invalid data')
            return np.nan
        window = window + np.asarray([0, 1])
        r, c = np.meshgrid(np.arange(*window), np.arange(*window), indexing='ij')
        hi = []
        lo = []
        for i in range(0, num_pts):  # pylint: disable=unused-variable
            iloc = np.nanargmax(data)
            hi.append(data[iloc])
            row, col = np.unravel_index(iloc, shape)  # pylint: disable=unbalanced-tuple-unpacking
            locs = np.ravel_multi_index(
                ((row + r).ravel(), (col + c).ravel()),
                shape, mode='clip')
            data[locs] = np.nan
            iloc = np.nanargmin(data)
            lo.append(data[iloc])
            row, col = np.unravel_index(iloc, shape)  # pylint: disable=unbalanced-tuple-unpacking
            locs = np.ravel_multi_index(
                ((row + r).ravel(), (col + c).ravel()),
                shape, mode='clip')
            data[locs] = np.nan
        rz = (np.nansum(hi) - np.nansum(lo)) / num_pts
        # del data
        return rz

    # ----z scale----
    def calc_colormap(self, **params):
        colorscale = Surface.ColorScale(self, **params)
        self.cmap = colorscale.cmap
        self.cmap_prms = colorscale.cmap_prms
        # print(colorscale)
        return self.cmap, self.cmap_prms

    class ColorScale:
        """
        base: 'ra', 'rq', 'pv', 'fixed'
        params: list
            if stats based:  'zfac' as stretching factor (<0 to force default)
            if manual scale: 'zmin, zmax, z_lo, z_hi'
        """

        def __init__(self, surface, cmap='turbo', **params):
            self.lut = cm.get_cmap(cmap)(np.linspace(0, 1, 256))
            self.surface = surface
            self.cmap_prms = {'cmap': cmap, 'base': params.get('base', 'ra')}
            self.cmap, new_params = self._update(**params)
            self.cmap_prms.update(new_params)

        def __str__(self):
            # return str(self.cmap_prms)
            string = 'ColorScale:'
            for key, value in self.cmap_prms.items():
                fmt = '0.3f'
                if isinstance(value, str):
                    fmt = 's'
                string = string + f' {key}={value:{fmt}}'
            return string

        def _update(self, base='ra', **params):
            if params.get('zfac', -1) < 0.001:  # force default
                params.pop('zfac', 2.0)
            new_params = getattr(self, f'_{base}_based')(**params)
            new_params = dict(zip(['zfac', 'zmin', 'zmax', 'z_lo', 'z_hi'], new_params))
            return self._calc_cmap(self.lut, **new_params), new_params

        def _ra_based(self, zfac=2.0, **kwargs):
            ra = self.surface.ra
            rq = self.surface.rq
            data = self.surface.automasking(3.0 * rq)
            median = np.nanmedian(data)
            z_lo = -zfac * ra + median
            z_hi = zfac * ra + median
            return zfac, np.nanmin(data), np.nanmax(data), z_lo, z_hi

        def _rq_based(self, zfac=3.0, **kwargs):
            rq = self.surface.rq
            data = self.surface.automasking(6.0 * rq)
            median = np.nanmedian(data)
            z_lo = -zfac * rq + median
            z_hi = zfac * rq + median
            return zfac, np.nanmin(data), np.nanmax(data), z_lo, z_hi

        def _pv_based(self, zfac=12.0, **kwargs):
            rq = self.surface.rq
            rv = self.surface.rv
            rp = self.surface.rp
            median = self.surface.median
            ratio = zfac * rq / (rp - rv)
            z_lo = -ratio * rq + median
            z_hi = ratio * rq + median
            return zfac, rv, rp, z_lo, z_hi

        def _fixed_based(self, zfac=3.0, **kwargs):
            if len(kwargs) == 4:
                return [zfac, ] + list(kwargs.values())
            zmin = np.negative(zfac)
            zmax = zfac
            median = self.surface.median
            z_lo = 0.40 * zmin + median
            z_hi = np.negative(z_lo)
            return zfac, zmin, zmax, z_lo, z_hi

        @staticmethod
        def _calc_cmap(cmap, zmin, zmax, z_lo, z_hi, **kwargs):
            num = 240
            z = (np.linspace(zmin, zmax, num=num, endpoint=True))
            z_lo1 = np.linspace(zmin, z_lo, num=15, endpoint=False)
            z_lo2 = np.linspace(z_lo, 0.0, num=105, endpoint=False)
            z_lo0 = np.append(z_lo1, z_lo2)
            z_hi1 = np.linspace(0.0, z_hi, num=85, endpoint=False)
            z_hi2 = np.linspace(z_hi, zmax, num=35, endpoint=True)
            z_hi0 = np.append(z_hi1, z_hi2)
            new_z = np.append(z_lo0, z_hi0)
            new_z = np.interp(z, new_z, z)
            idx = 0
            blue = cmap[idx:idx + num, 0]
            green = cmap[idx:idx + num, 1]
            red = cmap[idx:idx + num, 2]
            new_cmap = np.array([np.interp(new_z, z, blue),
                                 np.interp(new_z, z, green),
                                 np.interp(new_z, z, red)]).T
            return lsc.from_list('veeco', new_cmap, num)

    def plot(self, title=None, reverse='', **kwargs):
        """Matplotlib 2D plot with some additional functions.

            Parameters :
                'title' : string

                'reverse' : None,'x', 'y' or 'xy'  (default None).

                colormap parameters to set the colormap (see ColorScale class):
                    'base': 'ra', 'rq', 'pv', 'fixed'
                    'params': list
                        if stats based:  'zfac' as stretching factor
                        if manual scale: 'zmin, zmax, z_lo, z_hi'

                matplotlib.axes.Axes.imshow arguments
                https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
        """
        colorprms = {}
        for arg in ('base', 'cmap', 'zfac', 'zmin', 'zmax', 'z_lo', 'z_hi'):
            if arg in kwargs:
                colorprms[arg] = kwargs.pop(arg, None)
        self.calc_colormap(**colorprms)

        import matplotlib.pyplot as plt
        plt.matshow(
            self.z.T,
            origin=kwargs.get('origin', 'lower'),
            cmap=self.cmap,
            vmin=self.cmap_prms['zmin'], vmax=self.cmap_prms['zmax'],
            extent=[self.x.min(), self.x.max(),
                    self.y.min(), self.y.max()],
            **kwargs,
        )
        if title is not None:
            plt.title(title)
        plt.colorbar()
        if 'x' in reverse.lower():
            plt.gca().invert_xaxis()
        if 'y' in reverse.lower():
            plt.gca().invert_yaxis()
        plt.show()


# ------------generic data class------------
class Header:
    # def __repr__(self):
    #     return self
    def __str__(self):
        return 'Header block'


def _iter_header_dict(dico, parent):
    for key, value in dico.items():
        if isinstance(value, dict):
            child = Header()
            setattr(parent, key, child)
            _iter_header_dict(value, child)
        else:
            setattr(parent, key, value)


class ESRFOpticsLabData:
    """Common data class."""
    organisation = "ESRF"
    division = 'ISDD'
    group = "XOG"
    laboratory = "Mirror & Metrology Lab"

    def __init__(self):
        self.path = None
        self.source = None
        self.header = dict()
        self.initial = None
        self.analysis_steps = list()

        self.coords = None
        self.values = None
        self.mask = None
        self.units = {  # as example
            'coords': None, 'values': None,  # must be present
            'height': None, 'angle': None,
            'length': None, 'radius': None,
            'pixel': None, }

    @classmethod
    def read(cls, *args, verbose=False):
        self = cls().readfile(*args)
        if self is None:
            warnings.warn('Error reading data (No such file).')
            return None
        super(ESRFOpticsLabData, self).__init__(  # pylint: disable=super-with-arguments
            self.initial.coords, self.initial.values,
            self.initial.units, self.source)
        if verbose:
            print(f'  file \'{self.source}\' opened.')
        return self

    # ----methods to be overrided----
    def readfile(self, path, source=None):  # pylint: disable=unused-argument
        error_msg = f'{self!r}: \'readfile\' function must be overrided.'
        print(error_msg)
        return self

    # ----common methods----
    def header_dict_to_class(self):
        _iter_header_dict(self.header, self)

    # ----analysis----
    def add_analysis_step(self, funcname, *params):
        step = (funcname, params)
        self.analysis_steps.append(step)

    def analyze(self):
        for funcname, params in self.analysis_steps:
            print(f'      {self.source}: applying {funcname}')
            getattr(self, funcname)(*params)
