# coding=utf-8
class EllipseBase:
    """Base class for fitting tangential ellipse to an n-dimensional data"""
    def __init__(self, ellipse_params, checked_params=[0, 1, 1, 0, 0, 0]):
        """
        Initialize ellipse base class with inputs from OWFit widget. All parameters and datasets are in si units.

        :param ellipse_params: Ellipse parameters list with values p, q, theta, center_offset, rotate/tilt, piston
        :type ellipse_params: list[float]
        :param checked_params: Checked parameters used in optimization for parameters p, q, theta, center_offset, rotate/tilt, piston
        :type checked_params: list[int/bool]
        """
        self.ellipse_params = ellipse_params
        self.checked_params = checked_params

    def fit(self, dtype, x, data):
        """
        Implemented in subclasses

        :param dtype: Data type, 'slopes_x' or 'height'
        :type dtype: str
        :param x: X position vector with same length as data
        :type x: np.ndarray
        :param data: Data in slopes or height
        :type data: np.ndarray
        :return: Fitted parameters
        :rtype: list[float]
        """
        fitted_ellipse_params = None
        return fitted_ellipse_params

    def get_ellipse(self, dtype, x, ellipse_params):
        """
        Implemented in subclasses

        :param dtype: Data type, 'slopes_x' or 'height'
        :type dtype: str
        :return: Ellipse tangential slopes for given parameters
        :rtype: np.ndarray
        """
        data = None
        return data
