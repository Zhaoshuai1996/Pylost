import ast
import copy
import json
from collections import Sequence

from astropy.units import Quantity
from astropy import units as u
import numpy as np
from ast import literal_eval as make_tuple

from pylost_widgets.util.ufunc_metrology_data import get_params, insert_default_items, update_result


class MetrologyData(Quantity):
    """
    Class to comprehensively represent metrology data inherited from numpy ndarray and astropy quantity
    """
    _dim_detector = None
    _pix_size = 1.0 * u.dimensionless_unscaled
    _axis_names = None
    _init_shape = None
    _index_list = None
    _axis_values = None
    _motors = []
    _flags = {}
    _items = None

    def get_print_attributes(self):
        """
        Get MetrologyData parameters in printable format.

        :return: Printable parameters
        :rtype: dict
        """
        return {
            'unit': '{}'.format(self.unit),
            'pix_size': ' x '.join('{:.6f}'.format(x) for x in self.pix_size),
            'dim_detector': ' x '.join(str(x) for x in self.dim_detector),
            'axis_names': ' x '.join(self.axis_names),
            'init_shape': ' x '.join(str(x) for x in self.init_shape),
            'start_position': ' x '.join('{:.5f}'.format(x) for x in self.start_position),
            'flags': json.dumps(self._flags),
            'axis_values': ' x '.join('{}'.format(x) for x in self.axis_values),
            'motors': '\n\t\t\t'.join('{} : {} {}'.format(x['name'],
                                                          list(np.around(x['values'], 5)) if isinstance(x['values'],
                                                                                                        Sequence) else np.around(
                                                              x['values'], 5), x['unit']) for x in self.motors)
        }

    def get_attributes(self):
        """
        Get MetrologyData parameters in savable format as attributes in hdf5.

        :return: Parameters
        :rtype: dict
        """
        return {
            'unit': str(self.unit),
            'dim_detector': np.array(self.dim_detector),
            'pix_size': np.array([x.value for x in self.pix_size]),
            'pix_size_unit': json.dumps([str(x.unit) for x in self.pix_size]),
            'axis_names': json.dumps(self.axis_names),
            'init_shape': np.array(self.init_shape if self.init_shape is not None else self.shape),
            'index_list': str([list(x) for x in self.index_list]),
            'axis_values': json.dumps(
                [[list(x.values), x.unit] if isinstance(x, Quantity) else str(x) for x in self.axis_values]),
            'motors': json.dumps(
                [{k: (list(v) if any(v.shape) else v.item()) if isinstance(v, np.ndarray) else v for k, v in x.items()}
                 for x in self.motors]),
            'flags': json.dumps(self._flags)
        }

    @classmethod
    def apply_attributes(cls, data, attrs):
        """
        Apply MetrologyData from a dictionary

        :param data: Input data to add attributes
        :type data: np.ndarray
        :param attrs: Data attributes
        :type attrs: dict
        :return: Final data object
        :rtype: MetrologyData
        """
        unit = attrs['unit']
        pix_size = attrs['pix_size']
        if isinstance(pix_size, str) and len(pix_size.split(' x ')) > 1:
            pix_size = [ast.literal_eval(x) for x in pix_size.split(' x ')]
        if not isinstance(pix_size, (list, np.ndarray, tuple)):
            pix_size = [pix_size]
        pix_unit = json.loads(attrs['pix_size_unit']) if 'pix_size_unit' in attrs else [None] * len(pix_size)
        pix_size = [Quantity(x, unit=y) for x, y in zip(pix_size, pix_unit)]
        dim_detector = attrs['dim_detector'] if 'dim_detector' in attrs else [True] * len(pix_size)
        axis_names = json.loads(attrs['axis_names']) if 'axis_names' in attrs else None
        axis_values = json.loads(attrs['axis_values']) if 'axis_values' in attrs else None
        if axis_values is not None:
            axis_values = [Quantity(x[0], x[1]) if type(x) == list and len(x == 2) and type(x[1]) == str else x for x in
                           axis_values]
        motors = json.loads(attrs['motors']) if 'motors' in attrs else None
        flags = json.loads(attrs['flags']) if 'flags' in attrs else None
        data = MetrologyData(data, unit=unit, pix_size=pix_size, dim_detector=dim_detector, axis_names=axis_names,
                             axis_values=axis_values, motors=motors, flags=flags)
        try:
            init_shape = attrs['init_shape'] if 'init_shape' in attrs else None
            index_list = ast.literal_eval(attrs['index_list']) if 'index_list' in attrs else None
            data._set_init_shape(init_shape)
            data._set_index_list(index_list)
        except Exception as e:
            print(e)
        return data

    def _format_dim_detector(self, dim_detector):
        """
        Format detector dimensions object, e.g. [False, True, True].

        :param dim_detector: Detector dimensions
        :type dim_detector: tuple/list/np.ndarray
        :return: Formatted detector dimensions
        :rtype: list[bool]
        """
        if isinstance(dim_detector, (tuple, list, np.ndarray)) and len(dim_detector) == self.ndim \
                and all([type(x) is bool for x in dim_detector]):
            return list(dim_detector)
        retArr = np.asarray([False] * self.ndim)
        if dim_detector is not None:
            if type(dim_detector) in [bool, np.bool_]:
                dim_detector = [dim_detector]
            if isinstance(dim_detector, (tuple, list, np.ndarray)):
                for i, val in enumerate(dim_detector[::-1]):
                    if type(val) in [bool, np.bool_]:
                        j = -(i + 1)
                        if -len(retArr) <= j < len(retArr):
                            retArr[j] = val
                    elif -len(retArr) <= val < len(retArr):  # TODO: and type(val) is int
                        retArr[val] = True
        return list(retArr)

    def _format_index_list(self, index_list):
        """
        Format indices list along all data dimensions, e.g. [[0, 1, 2], [25, 26, 27, 28, 29, 30], [35, 36, 37, 38, 39]].

        :param index_list: List of arrays, each item is a array of indices along a dimension
        :type index_list: list
        :return: Formatted list
        :rtype: list[np.ndarray]
        """
        try:
            dim_nonzero = np.array(self._dim_detector).nonzero()[0]
            if index_list is None or not isinstance(index_list, list):
                return [np.arange(x) for x in self.shape]
            elif len(index_list) == self.ndim:
                for i, x in enumerate(self.shape):
                    if type(index_list[i]) is list:
                        index_list[i] = np.array(index_list[i])
                    if type(index_list[i]) is not np.ndarray:
                        index_list[i] = np.arange(x)
                    elif len(index_list[i]) != x:
                        index_list[i] = np.arange(x)
                return index_list
            elif len(index_list) == len(dim_nonzero):
                a = [np.arange(x) for x in self.shape]
                i = 0
                for x in dim_nonzero:
                    a[x] = np.array(index_list[i])
                    i += 1
                return a
            elif len(index_list) > self.ndim:
                return index_list[-self.ndim:]
            elif len(index_list) < self.ndim:
                a = [np.arange(x) for x in self.shape[:(self.ndim - len(index_list))]]
                return a + index_list

        except Exception as e:
            print('Exception while formatting MetrologyData index list')
            print(e)

    def _format_axis_names(self, axisNames):
        """
        Format axis names for all dimensions, e.g. [Motor, Y, X].

        :param axisNames: Axis names array
        :type axisNames: tuple/list/np.ndarray
        :return: Formatted axis names
        :rtype: list[str]
        """
        if isinstance(axisNames, (tuple, list, np.ndarray)) and len(axisNames) == self.ndim:
            return list(axisNames)
        retArr = [''] * self.ndim
        if axisNames is not None and isinstance(axisNames, (list, tuple, np.ndarray)):
            for i, item in enumerate(axisNames[::-1]):
                j = -(i + 1)
                if -len(retArr) <= j < len(retArr):
                    retArr[j] = item
        return retArr

    def _format_axis_values(self, axisVals):
        """
        Format axis values, e..g ['motor_X', '', ''] where motor_X name exists in self._motors.

        :param axisVals: Axis values array
        :type axisVals: tuple/list/np.ndarray
        :return: Formatted axis values
        :rtype: list[str/np.ndarray]
        """
        if isinstance(axisVals, (tuple, list, np.ndarray)) and len(axisVals) == self.ndim:
            return list(axisVals)
        retArr = [''] * self.ndim
        if axisVals is not None and isinstance(axisVals, (list, tuple, np.ndarray)):
            for i, item in enumerate(axisVals[::-1]):
                j = -(i + 1)
                if -len(retArr) <= j < len(retArr):
                    retArr[j] = item
        return retArr

    def _format_motors(self, motors):
        """
        Format motors list. Each item is a dictionary with motor name, values, unit and axis fields.

        :param motors: Motors array
        :type motors: tuple/list/np.ndarray
        :return: Formatted motors array
        :rtype: list[dict]
        """
        if isinstance(motors, (tuple, list, np.ndarray)) and all([type(x) is dict for x in motors]):
            return list(motors)
        elif isinstance(motors, dict):
            return [motors]
        else:
            return []

    def _format_flags(self, flags):
        """
        Format flags dictionary, e.g. {flip_y_display:True}.

        :param flags: Flags dictionary
        :type flags: dict
        :return: Formatted flags
        :rtype: dict
        """
        if isinstance(flags, dict):
            return flags
        else:
            return {}

    def _format_init_shape(self, init_shape):
        """
        Format initial shape, before applying any masks.

        :param init_shape: Initial shape array
        :type init_shape: tuple/list/np.ndarray
        :return: Formatted init shape
        :rtype: list[int]
        """
        if isinstance(init_shape, (tuple, list, np.ndarray)) and len(init_shape) == self.ndim:
            return list(init_shape)
        retArr = list(self.shape)
        if init_shape is not None and isinstance(init_shape, (list, tuple, np.ndarray)):
            for i, item in enumerate(init_shape[::-1]):
                j = -(i + 1)
                if -len(retArr) <= j < len(retArr):
                    retArr[j] = item
        return retArr

    def _format_pix_sz(self, pix_size, unit=None):
        """
        Format pixel size array. It is set to 1 for non-detector dimensions.

        :param pix_size: Pixel size
        :type pix_size: tuple/list/np.ndarray
        :param unit: Pixel units
        :type unit: str/astropy.units.Unit
        :return: Formatted pixel size array
        :rtype: list[Quantity]
        """
        if isinstance(pix_size, Quantity) and np.isscalar(pix_size.value):
            pix_size = [pix_size]
        if isinstance(pix_size, (tuple, list, np.ndarray)) and len(pix_size) == self.ndim \
                and all([type(x) is Quantity for x in pix_size]):
            return list(pix_size)

        retArr = [1.0 * u.dimensionless_unscaled] * self.ndim
        retArr = np.asarray(retArr, dtype=Quantity)
        if pix_size is not None:
            if self.dim_detector is None:
                raise Exception('Dimensions of detector are not selected')

            pix_size = self.get_pix_sz_quantity(pix_size, unit=unit)
            retArr[self.dim_detector] = pix_size
        return list(retArr)

    def get_pix_sz_quantity(self, pix_size, unit=None):
        """
        Get pixel size as list of astropy quantities.

        :param pix_size: pixel size array
        :type pix_size: tuple/list/np.ndarray
        :param unit: pixel unit
        :type unit: str/astropy.units.Unit
        :return: Pixel size as quantities
        :rtype: list[Quantity]
        """
        retArr = []
        if isinstance(pix_size, (list, tuple, np.ndarray)):
            for item in pix_size:
                retArr.append(Quantity(item, unit=unit))
        elif np.issubdtype(type(pix_size), np.number):
            retArr.append(Quantity(np.double(pix_size), unit=unit))
        elif isinstance(pix_size, Quantity):
            retArr.append(pix_size)
        elif isinstance(pix_size, str):
            try:
                retArr.append(Quantity(np.double(pix_size), unit=unit))
            except:
                try:
                    retArr.append(self.get_pix_sz_quantity(make_tuple(pix_size), unit=unit))
                except:
                    raise ValueError('Unable to parse string')
        return retArr

    def get_pix_scale(self):
        """
        Get pixel scale, i.e. scale when units are converted to SI.

        :return: Pixel scale
        :rtype: list[float]
        """
        axis_vals = self.get_axis_val_items_detector()
        if self.dim_detector is not None and np.any(
                self.dim_detector) and False in self.dim_detector:  # only detector data
            pix_scale = [axis_vals[i].unit.si.scale for i, x in enumerate(axis_vals)]
        else:  # last one or two dimensions depending on ndim
            if self.ndim == 1:
                pix_scale = [axis_vals[-1].unit.to('m')]  # .si.scale]
            else:
                pix_scale = [axis_vals[i].unit.to('m') for i in [-2, -1]]  # .si.scale
        return pix_scale

    def get_axis_units_detector(self):
        """
        Get axis units only along detector dimensions. Useful if axis values are specified instead if pixel size.

        :return: Units
        :rtype: astropy.units.Unit
        """
        axis_vals = self.get_axis_val_items_detector()
        if self.dim_detector is not None and np.any(
                self.dim_detector) and False in self.dim_detector:  # only detector data
            units = [axis_vals[i].unit for i, x in enumerate(axis_vals)]
        else:  # last one or two dimensions depending on ndim
            if self.ndim == 1:
                units = [axis_vals[-1].unit]
            else:
                units = [axis_vals[i].unit for i in [-2, -1]]
        return units

    def get_axis_val_items_detector(self):
        """
        Get axis values along detector dimensions.

        :return: Axis values
        :rtype: list[Quantity[np.ndarray]]
        """
        axis_vals = self.get_axis_val_items()
        if self.dim_detector is not None and np.any(self.dim_detector) and False in self.dim_detector:
            axis_vals = [axis_vals[i] for i in np.where(self.dim_detector)[0]]
        return axis_vals

    def get_axis_val_items(self):
        """
        Get axis values for all dimensions. If an item is string look for corresponding values in self._motors. If for certain detector axis no values are provided, build using pixel size.

        :return: Axis values
        :rtype: list[Quantity[np.ndarray]]
        """
        retArr = [Quantity([])] * self.ndim
        try:
            if self._axis_values is not None:
                for i, item in enumerate(self._axis_values):
                    if isinstance(item, str) and item != '':
                        for m in self._motors:
                            if m['name'] == item:
                                temp = m['values']
                                if len(temp) == self.shape[i]:  # Dimension size is same as motor array size
                                    retArr[i] = Quantity(temp, unit=m['unit'])
                                # elif np.prod(self.shape_non_detector) == len(flatten_list(temp)): # Each motor position correspond to full non detector size
                                #     pass
                                elif self._index_list is not None and len(temp) > np.max(self._index_list[i]) and len(
                                        self._index_list[i]) == self.shape[i]:
                                    retArr[i] = Quantity(np.asarray(temp)[self._index_list[i]], unit=m['unit'])
                    elif isinstance(item, Quantity) and isinstance(item.value, np.ndarray):
                        if len(item) == self.shape[i]:
                            retArr[i] = item
                        elif self._index_list is not None and len(item) > np.max(self._index_list[i]) and len(
                                self._index_list[i]) == self.shape[i]:
                            retArr[i] = item[self._index_list[i]]
                    elif self.dim_detector is not None and self.dim_detector[i] and self._pix_size is not None:
                        #     if self._index_list is not None and len(self._index_list[i]) == self.shape[i]:
                        #         retArr[i] = self._pix_size[i] * self._index_list[i]
                        #     else:
                        retArr[i] = self._pix_size[i] * np.arange(self.shape[i])
                        # retArr[i] = retArr[i] - np.nanmean(retArr[i])

        except Exception as e:
            print(e)
        return retArr

    @property
    def dim_detector(self):
        """Detector dimensions boolean array"""
        return self._dim_detector

    @property
    def pix_size(self):
        """Pixel size array"""
        return self._pix_size

    @property
    def pix_size_detector(self):
        """Pixel sizes for detector dimensions"""
        if self._dim_detector is not None:
            return np.asarray(self._pix_size, dtype=Quantity)[self._dim_detector]
        else:
            return self._pix_size

    @property
    def init_shape(self):
        """Initial data shape"""
        return self._init_shape

    @property
    def index_list(self):
        """Indices list"""
        return self._index_list

    @property
    def axis_names(self):
        """Axis names list"""
        return self._axis_names

    @property
    def axis_values(self):
        """Axis values list"""
        return self._axis_values

    @property
    def axis_names_detector(self):
        """Axis names for detector dimensions"""
        if self._dim_detector is not None:
            return np.asarray(self._axis_names, dtype=Quantity)[self._dim_detector]
        else:
            return self._axis_names

    @property
    def index_list_detector(self):
        """Indices list for detector dimensions"""
        if self._dim_detector is not None and self._index_list is not None:
            return [self.index_list[i] for i in np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def start_position(self):
        """Start position of data on detector (in detector units)"""
        if self._dim_detector is not None and self._index_list is not None:
            return [self.index_list[i][0] * self.pix_size[i] for i in np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def start_position_pix(self):
        """Start position of data on detector (in pixels)"""
        if self._dim_detector is not None and self._index_list is not None:
            return [self.index_list[i][0] for i in np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def center(self):
        """Center position relative to data size (in detector units)"""
        if self._dim_detector is not None and self._index_list is not None:
            return [(self.index_list[i][0] + 0.5 * self.shape[i]) * self.pix_size[i] for i in
                    np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def center_pix(self):
        """Center position relative to data size (in pixels)"""
        if self._dim_detector is not None and self._index_list is not None:
            return [int(self.index_list[i][0]) + int(0.5 * self.shape[i]) for i in np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def center_absolute(self):
        """Center position relative to initial/full size (in detector dimensions)"""
        if self._dim_detector is not None and self._init_shape is not None:
            return [self._init_shape[i] * self.pix_size[i] / 2 for i in np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def center_absolute_pix(self):
        """Center position relative to initial/full size (in pixels)"""
        if self._dim_detector is not None and self._init_shape is not None:
            return [int(self._init_shape[i] / 2) for i in np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def size_init_detector(self):
        """Initial size of detector"""
        if self._dim_detector is not None and self._init_shape is not None:
            return [self._init_shape[i] * self.pix_size[i] for i in np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def size_detector(self):
        """Current size of detector data"""
        if self._dim_detector is not None:
            return [self.shape[i] * self.pix_size[i] for i in np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def shape_detector(self):
        """Current size of detector data"""
        if self._dim_detector is not None:
            return [self.shape[i] for i in np.where(self.dim_detector)[0]]
        else:
            return None

    @property
    def shape_non_detector(self):
        """Data shape along non-detector dimensions"""
        if self._dim_detector is not None:
            return [self.shape[i] for i in np.where(np.invert(self.dim_detector))[0]]
        else:
            return None

    @property
    def motors(self):
        """Motor list"""
        return self._motors

    def _set_dim_detector(self, dim_detector):
        self._dim_detector = self._format_dim_detector(dim_detector)

    def _set_pix_size(self, pix_size, unit=None):
        self._pix_size = self._format_pix_sz(pix_size, unit=unit)

    def _set_init_shape(self, _init_shape):
        self._init_shape = self._format_init_shape(_init_shape)

    def _set_index_list(self, index_list):
        self._index_list = self._format_index_list(index_list)

    def _set_axis_names(self, axis_names):
        self._axis_names = self._format_axis_names(axis_names)

    def _set_axis_values(self, axis_vals):
        self._axis_values = self._format_axis_values(axis_vals)

    def _set_motors(self, motors):
        self._motors = self._format_motors(motors)

    def _set_flags(self, flags):
        self._flags = self._format_flags(flags)

    def _set_start_position(self, new_start_pos):
        """
        To update start position, indices list has to be updated, i.e. start pixel is added to index list along each detector dimension.

        :param new_start_pos: New start position
        :type new_start_pos: list
        :return: Flag whether the new position is set
        :rtype: bool
        """
        pidx = np.zeros(self.ndim, dtype=int)
        pidx[self.dim_detector] = self.start_position_pix
        nidx = np.zeros(self.ndim, dtype=int)
        narr = np.asarray(new_start_pos, dtype=object)
        if not np.all([x.unit == '' for x in narr]):
            narr = narr / self.pix_size_detector
        nidx_det = [np.ceil(x.to('').value) for x in narr]
        nidx[self.dim_detector] = nidx_det
        size = np.asarray(self.shape)[self.dim_detector]
        full_size = np.asarray(self.init_shape)[self.dim_detector]
        if np.any(np.asarray(nidx_det + size) > np.asarray(full_size)):
            return False
        else:
            idx_list = self.index_list
            for i in range(self.ndim):
                idx_list[i] = [x + nidx[i] - pidx[i] for x in self.index_list[i]]
            self._set_index_list(idx_list)
            return True

    def update_motors(self, motors, add_if_not=True):
        """
        Update motors array

        :param motors: Array with new/updated motor values
        :type motors: list[dict]
        :param add_if_not: Add if motor not existing
        :type add_if_not: bool
        """
        for m in motors:
            self.update_motor(m['name'], m['values'], m['unit'], m['axis'], add_if_not=add_if_not)

    def update_motor(self, name, values, unit, axis=-3, add_if_not=True):
        """
        Update a motor

        :param name: Name of motor
        :type name: str
        :param values: Motor values
        :type values: np.ndarray
        :param unit: Motor unit
        :type unit: str
        :param axis: Associated data axis
        :type axis: int
        :param add_if_not: Add motor if not existing
        :type add_if_not: bool
        """
        has_name = False
        for m in self._motors:
            if m['name'] == name:
                has_name = True
                m['values'] = values
                m['unit'] = '{}'.format(unit)
        if not has_name and add_if_not:
            self.add_motor(name, values, unit, axis=axis)

    def add_motor(self, name, values, unit, axis=-3):
        """
        Add a new motor

        :param name: Name of motor
        :type name: str
        :param values: Motor values
        :type values: np.ndarray
        :param unit: Motor unit
        :type unit: str
        :param axis: Associated data axis
        :type axis: int
        """
        m = {'name': name,
             'values': values,
             'axis': axis,
             'unit': '{}'.format(unit)}
        self._motors.append(m)

    def add_flag(self, name, val):
        """
        Add a new flag.

        :param name: Name of the flag
        :type name: str
        :param val: True or false
        :type val: bool
        """
        self._flags[name] = bool(val)

    def del_flag(self, name):
        """
        Delete a flag.

        :param name: Name of the flag
        :type name: str
        """
        del self._flags[name]

    def get_flag(self, name):
        """
        Get flag value.

        :param name: Name of the flag
        :type name: str
        :return: Flag value
        :rtype: bool
        """
        if self.has_flag(name):
            return self._flags[name]
        else:
            return False

    def has_flag(self, name):
        """
        Check if MetrologyData has the flag

        :param name: Name of the flag
        :type name: str
        :return: Has the flag or not
        :rtype: bool
        """
        return (name in self._flags)

    # Base class methods
    def __new__(cls, value, unit=None, pix_size=1.0, pix_unit=None, dim_detector=None, axis_names=None,
                axis_values=None, motors=None, flags=None, subok=True, **kwargs):
        """
        Create a new MetrologyData object and initialize it.

        :param value: Value stored in MetrologyData
        :type value: np.ndarray
        :param unit: Units
        :type unit: str/astropy.units.Unit
        :param pix_size: Pixel size
        :type pix_size: float/list
        :param pix_unit: Pixel unit
        :type pix_unit: str
        :param dim_detector: Detector dimensions
        :type dim_detector: list[bool]
        :param axis_names: Axis names
        :type axis_names: list[str]
        :param axis_values: Axis values
        :type axis_values: list[str/np.ndarray]
        :param motors: Motors list
        :type motors: list[dict]
        :param flags: Flags
        :type flags: dict
        :param subok: Used by astropy.Quantity class
        :type subok: bool
        :param kwargs: Additional arguments used by astropy.Quantity class
        :type kwargs: bool
        :return: New data object
        :rtype: MetrologyData
        """
        if isinstance(value, (Quantity, MetrologyData)):
            unit = value.unit
            value = value.value
        quantity = super().__new__(cls, value, unit=unit, subok=subok, **kwargs)
        quantity._init_shape = quantity.shape
        quantity._set_axis_names(axis_names)
        quantity._set_axis_values(axis_values)
        quantity._set_dim_detector(dim_detector)
        quantity._set_pix_size(pix_size, unit=pix_unit)
        quantity._set_index_list([np.arange(x) for x in quantity.shape])
        quantity._set_motors(motors)
        quantity._set_flags(flags)
        return quantity

    def __array_finalize__(self, obj):
        """
        Finalize function after numpy operation.

        :param obj: Data object
        :type obj: MetrologyData
        """
        # If we're a new object or viewing an Quantity, nothing has to be done.
        if obj is None or obj.__class__ is Quantity:
            return
        if self._unit is None:
            self._unit = getattr(obj, '_unit', None)
        self._dim_detector = getattr(obj, '_dim_detector', self._dim_detector)
        self._pix_size = getattr(obj, '_pix_size', self._pix_size)
        self._init_shape = getattr(obj, '_init_shape', self._init_shape)
        self._axis_names = getattr(obj, '_axis_names', self._axis_names)
        self._axis_values = getattr(obj, '_axis_values', self._axis_values)
        self._index_list = getattr(obj, '_index_list', self._index_list)
        self._motors = getattr(obj, '_motors', self._motors)
        self._flags = getattr(obj, '_flags', self._flags)
        self._items = getattr(obj, '_items', None)
        try:
            if self._items is not None:
                items = self._items
                self._copy_items()
                if not isinstance(items, MetrologyData):
                    try:  # TODO: mute some warnings temporary, are these errors meaningful?
                        self._update_slice(items)
                    except:
                        pass
        except Exception as e:
            print('MetrologyData-->__array_finalize__:')
            print(e)

    def __getitem__(self, items):
        self._items = items
        return super().__getitem__(items)

    def _update_slice(self, items):
        """
        Update metrology data parameters after a slicing operation.

        :param items: Slicing items
        :type items: bool / np.ndarray / tuple / list
        """
        if isinstance(items, bool):
            items = list(self.nonzero())
            items[0] = None
        if isinstance(items, np.ndarray) and items.dtype == bool:
            items = items.nonzero()
        if isinstance(items, tuple) and any(isinstance(x, bool) for x in items):
            non_items = [x for x in items if type(x) is bool]
            items = tuple([x for x in items if type(x) is not bool])
            if np.any(non_items) and len(items) == 0:
                items = list(self.nonzero())
                items[0] = None
            if not np.any(non_items):
                items = [np.array([])] * len(self._index_list)
        if isinstance(items, list) and any(isinstance(x, bool) for x in items):
            items = np.asarray(items).nonzero()
        if not isinstance(items, (tuple, np.ndarray, list)):
            items = (items,)
        if isinstance(items, (np.ndarray, list)) and all(np.issubdtype(type(x), np.integer) for x in items):
            items = (items,)

        terms = 0
        for i, item in enumerate(items):
            if item is Ellipsis:
                terms = len(self._index_list) - len(items)
                continue
            elif item is None:
                self._insert_default_items(i)
                continue
            if isinstance(item, (tuple, np.ndarray, list)):
                if isinstance(item, np.ndarray) and item.dtype == bool:
                    item = item.nonzero()
                elif isinstance(item, (tuple, np.ndarray, list)) and any(isinstance(x, bool) for x in item):
                    item = item.nonzero()
                self._index_list[i + terms] = item[0] if len(item) == 1 else list(np.unique(item))
            else:
                self._index_list[i + terms] = self._index_list[i + terms][item]

        if len(self.shape) < len(self._index_list):
            del_idx = []
            for i, item in enumerate(self._index_list):
                if type(item) is not np.ndarray:
                    del_idx.append(i)
            self._delete_items(del_idx)

        self._items = None

    def _insert_default_items(self, i):
        """
        Insert default parameters along a given axis.

        :param i: Axis position
        :type i: int
        """
        self._index_list.insert(i, [0])
        self._dim_detector.insert(i, False)
        self._pix_size.insert(i, Quantity(1.0))
        self._axis_names.insert(i, '')
        self._axis_values.insert(i, '')
        self._init_shape.insert(i, 1)

    def _delete_items(self, idx):
        """
        Delete parameters along given axis.

        :param idx: Axis position
        :type idx: int
        """
        self._index_list = self._del_item(self._index_list, idx)
        self._dim_detector = self._del_item(self._dim_detector, idx)
        self._pix_size = self._del_item(self._pix_size, idx)
        self._axis_names = self._del_item(self._axis_names, idx)
        self._axis_values = self._del_item(self._axis_values, idx)
        self._init_shape = self._del_item(self._init_shape, idx)

    def _copy_items(self):
        """
        Deep copy metrology data parameters
        """
        self._index_list = copy.deepcopy(self._index_list)
        self._dim_detector = copy.deepcopy(self._dim_detector)
        self._pix_size = copy.deepcopy(self._pix_size)
        self._axis_names = copy.deepcopy(self._axis_names)
        self._axis_values = copy.deepcopy(self._axis_values)
        self._init_shape = copy.deepcopy(self._init_shape)
        self._motors = copy.deepcopy(self._motors)
        self._flags = copy.deepcopy(self._flags)

    def _update_items(self, i, indices, is_dim_detector, pix_size, axis):
        """
        Update parameters along a given axis.

        :param i: Axis index
        :type i: int
        :param indices: Indices for the axis
        :type indices:
        :param is_dim_detector: Is the axes for detector dimension
        :type is_dim_detector: bool
        :param pix_size: Pixel size
        :type pix_size: float/Quantity[float]
        :param axis: Axis
        :type axis: int
        """
        self._index_list[i] = indices
        self._dim_detector[i] = is_dim_detector
        self._pix_size[i] = Quantity(pix_size)
        self._axis_names[i] = axis

    def _del_item(self, param, idx):
        """
        Delete parameter values along given axis

        :param param: Parameter values to update
        :type param: list
        :param idx: Axis
        :type idx: int
        :return: Updated parameter values
        :rtype: list
        """
        if param is None:
            return param
        else:
            return [item for i, item in enumerate(param) if i not in idx]

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        """
        Function called after array_finalize in a numpy operation. Final array is converted to MetrologyData

        :param function: Numpy operation/function which was applied to data
        :type function: Callable
        :param method:
        :type method:
        :param inputs: Inputs to the function
        :type inputs: list
        :param kwargs: Additional arguments
        :type kwargs: dict
        :return: Result as a MetrologyData
        :rtype: MetrologyData
        """
        # arrays = []
        # for input_ in inputs:
        #     arrays.append(input_.view(Quantity) if type(input_) is MetrologyData else input_)
        result = super().__array_ufunc__(function, method, *inputs, **kwargs)
        if result is None or result is NotImplemented:
            return result
        return self.result_metrology(result, function, method, *inputs, **kwargs)

    def result_metrology(self, result, function, method, *inputs, **kwargs):
        try:
            if isinstance(result, (tuple, list)):
                result_new = []
                for res in result:
                    res = self.result_metrology(res, function, method, *inputs, **kwargs)
                    result_new.append(res)
                return result.__class__(result_new)

            if isinstance(result, (MetrologyData, Quantity)):
                if type(result) is Quantity:
                    result = result.view(MetrologyData)
                nin = len(inputs)  # function.nin
                ndim = inputs[0].ndim
                params = get_params(inputs[0])
                if nin > 1:
                    for i in range(1, nin):
                        if isinstance(inputs[i], MetrologyData):
                            if inputs[i].ndim > ndim:
                                ndim = inputs[i].ndim
                    if ndim > inputs[0].ndim:
                        for i in range(ndim - inputs[0].ndim):
                            insert_default_items(0, params)

                return update_result(result, params, function, method, *inputs, **kwargs)
            else:
                return result
        except Exception as e:
            print('MetrologyData-->result_metrology:')
            print(e)
            return result

    def __quantity_subclass__(self, unit):
        """
        Specifies the current class as Quantity subclass

        :param unit: Units
        :type unit: str/astropy.units.Unit
        :return: MetrologyData class, True
        :rtype: Class, bool
        """
        return MetrologyData, True

    def copy_to(self, obj):
        """
        Extend current numpy array with paramters from Metrology Data object.

        :param obj: Original object
        :type obj: np.ndarray
        :return: Extended object
        :rtype: MetrologyData
        """
        if obj is None:
            return obj
        unit = obj.unit if type(obj) is Quantity else self.unit
        if isinstance(obj, np.ndarray):
            obj = obj.view(MetrologyData)
            obj._set_unit(unit)
            obj._set_dim_detector(self._dim_detector)
            obj._set_pix_size(self._pix_size)
            obj._set_index_list(copy.deepcopy(self._index_list))
            obj._set_axis_names(self._axis_names)
            obj._set_axis_values(self._axis_values)
            obj._set_init_shape(self._init_shape)
            obj._set_motors(copy.deepcopy(self._motors))
            obj._set_flags(self._flags)
        return obj

    def copy_to_detector_format(self, obj):
        """
        Extend current numpy array with paramters from Metrology Data object for only detector dimensions.

        :param obj: Original object
        :type obj: np.ndarray
        :return: Extended object
        :rtype: MetrologyData
        """
        if obj is None:
            return obj
        unit = obj.unit if type(obj) is Quantity else self.unit
        if isinstance(obj, np.ndarray):
            obj = obj.view(MetrologyData)
            obj._set_unit(unit)
            obj._set_dim_detector([x for x in self._dim_detector if x])
            obj._set_pix_size(self.pix_size_detector)
            obj._set_index_list(None)
            obj._set_axis_names(self.axis_names_detector)
            obj._set_axis_values(None)
            obj._set_motors(copy.deepcopy(self._motors))
            obj._set_flags(self._flags)
        return obj
