# coding=utf-8
'''
Created on Jul 16, 2018

@author: adapa
'''
from ctypes import *
import numpy as np

SHARPER = 'SHP'
FIZEAU = 'FIZ'
LTP = 'LTP'
MSI = 'MSI'


class uint2D(Structure):
    _fields_ = [
        ("X", c_uint),
        ("Y", c_uint)]


class int2D(Structure):
    _fields_ = [
        ("X", c_int),
        ("Y", c_int)]


class float2D(Structure):
    _fields_ = [
        ("X", c_float),
        ("Y", c_float)]


c_int_p = POINTER(c_int)
c_byte_p = POINTER(c_byte)
c_ubyte_p = POINTER(c_ubyte)
c_float_p = POINTER(c_float)
c_float_pp = POINTER(c_float_p)
c_long_p = POINTER(c_long)
c_ulong_p = POINTER(c_ulong)


def print_mat(A):
    print('\n'.join([' '.join(['{:4}'.format(item) for item in row])
                     for row in A]))


def ct_toMat(A):
    A_mat = [[val for val in row] for row in A]
    return A_mat


def ct_toList(A):
    A_list = [val for val in A]
    return A_list


def c2arr_to_nparr(c2arr):
    nparr = np.array([[x for x in row] for row in c2arr])
    return nparr


def carr_to_nparr(carr):
    nparr = np.array([x for x in carr])
    return nparr


def alertMsg(title, msg):
    import PyQt5.Qt as qt
    try:
        app = qt.QApplication.instance()
        if app is None:
            app = qt.QApplication([])
    except Exception as e:
        print('alertMsg <- commons')
        print(e)
    qt.QMessageBox.warning(None, title, msg)


def infoMsg(title, msg):
    import PyQt5.Qt as qt
    try:
        app = qt.QApplication.instance()
        if app is None:
            app = qt.QApplication([])
    except Exception as e:
        print('infoMsg <- commons')
        print(e)
    qt.QMessageBox.information(None, title, msg)


def questionMsg(parent=None, title="Title", msg="Yes/No?"):
    # from PyQt5.QtGui import QMessageBox
    from PyQt5.QtWidgets import QMessageBox
    answer = QMessageBox.question(parent,
                                  title,
                                  msg,
                                  QMessageBox.Yes | QMessageBox.No)
    if answer == QMessageBox.Yes:
        return True
    else:
        return False
