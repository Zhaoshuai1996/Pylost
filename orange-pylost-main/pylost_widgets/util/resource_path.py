# coding=utf-8
import os

import pylost_widgets


def resource_path(relative_path):
    """
    Get absolute path to resource.

    :param relative_path: Relative path
    :type relative_path: str
    :return: OS path of the given relative path
    :rtype: str
    """
    base_path = os.path.abspath(os.path.dirname(pylost_widgets.__file__))
    returnValue = os.path.join(base_path, relative_path)
    if not os.path.exists(returnValue):
        raise IOError("File %s dos not exists" % returnValue)
    return returnValue
