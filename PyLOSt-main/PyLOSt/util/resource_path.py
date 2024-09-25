# coding=utf-8
import os

import PyLOSt


def resource_path(relative_path):
    """ Get absolute path to resource """
    base_path = os.path.abspath(os.path.dirname(PyLOSt.__file__))
    returnValue = os.path.join(base_path, relative_path)
    if not os.path.exists(returnValue):
        raise IOError("File %s dos not exists" % returnValue)
    return returnValue
