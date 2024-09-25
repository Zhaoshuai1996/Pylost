# coding=utf-8
"""
Functions associated with MetrologyData.
"""
from astropy.units import Quantity
import copy
import numpy as np


def update_result(result, params, function, method, *inputs, **kwargs):
    """
    Update result MetrologyData array attributes (e.g. pixel size), from a numpy operation (e.g. numpy.mean).

    :param result: Result MetrologyData array
    :type result: MetrologyData
    :param params: New attributes for the result
    :type params: tuple
    :param function: Operation ufunc
    :type function: Callable
    :param method: Ufunc method: ``__call__``, ``at``, ``reduce``, etc.
    :type method: Callable
    :param inputs: Input arrays
    :type inputs: list
    :param kwargs: Additional arguments
    :type kwargs: dict
    :return: Updated result array
    :rtype: MetrologyData
    """
    if result.ndim != inputs[0].ndim:
        axis = kwargs.get('axis', None)
        if axis is not None:
            params = delete_items(axis, params)

    return update_result_params(result, params)


def update_result_params(result, params):
    """
    Update result array attributes.

    :param result: Result MetrologyData array
    :type result: MetrologyData
    :param params: Attributes of result array
    :type params: tuple
    :return: Updated result
    :rtype: MetrologyData
    """
    result._set_index_list(params[0])
    result._set_dim_detector(params[1])
    result._set_pix_size(params[2])
    result._set_axis_names(params[3])
    result._set_axis_values(params[4])
    result._set_init_shape(params[5])
    return result


def get_params(input, copyTo=True):
    """
    Get attributes from an input MetrologyData array.

    :param input: Input array
    :type input: MetrologyData
    :param copyTo: Flag to create deep copy of attributes
    :type copyTo: bool
    :return: index list, detector dimensions, pixel size, axis names, axis values, initial shape
    :rtype: tuple(list, list, list, list, list, list)
    """
    index_list = getattr(input, 'index_list', None)
    dim_detector = getattr(input, 'dim_detector', None)
    pix_size = getattr(input, 'pix_size', None)
    axis_names = getattr(input, 'axis_names', None)
    axis_values = getattr(input, 'axis_values', None)
    init_shape = getattr(input, 'init_shape', None)
    if copyTo:
        index_list = copy.deepcopy(index_list)
        dim_detector = copy.deepcopy(dim_detector)
        pix_size = copy.deepcopy(pix_size)
        axis_names = copy.deepcopy(axis_names)
        axis_values = copy.deepcopy(axis_values)
        init_shape = copy.deepcopy(init_shape)
    return (index_list, dim_detector, pix_size, axis_names, axis_values, init_shape)


def insert_default_items(i, params):
    """
    Insert default values for each attribute. Typically attributes are lists with length equivalent to number of data dimensions.
    Useful when the data dimensions are expanded.

    :param i: List index used for insertion
    :type i: int
    :param params: Data attributes
    :type params: tuple
    """
    params[0].insert(i, [0])
    params[1].insert(i, False)
    params[2].insert(i, Quantity(1.0))
    params[3].insert(i, '')
    params[4].insert(i, '')
    params[5].insert(i, 1)


def delete_items(idx, params):
    """
    Delete all attributes at an index. Useful when data dimensions are reduced.

    :param idx: Index
    :type idx: int
    :param params: Attributes list
    :type params: list
    :return: Updated parameters
    :rtype: tuple
    """
    params = list(params)
    for i in range(len(params)):
        params[i] = del_item(idx, params[i])
    return tuple(params)


def del_item(idx, param):
    """
    Delete selected attribute at the index.

    :param idx: Index or indices
    :type idx: int / list[int]
    :param param: All parameters
    :type param: tuple
    :return: Updated parameters
    :rtype: list
    """
    if not isinstance(idx, (list, tuple, np.ndarray)):
        if np.issubdtype(type(idx), np.integer):
            idx = [idx]
        else:
            raise ValueError('Index is not integer')
    return [item for i, item in enumerate(param) if i not in idx]
