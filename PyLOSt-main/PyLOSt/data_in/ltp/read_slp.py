# coding=utf-8
'''
Created on Dec 17, 2018

@author: adapa
'''
import numpy as np


def readSlp2File(fnm):
    """
    Reads .slp2 file

    :param fnm: slp2 file path
    :return: Raw data as dictionary
    """
    f = open(fnm, 'r')
    d = {}
    ds = []
    isSlpData = False
    for line in f:
        line = line.strip()
        if isSlpData:
            if line == '"':
                d['Slopes_Xmin'] = np.asarray(ds, dtype=float)
                try:
                    d['slopes_x'] = d['Slopes_Xmin'][:, 1]
                    d['motor_X'] = d['Slopes_Xmin'][:, 0]
                except Exception:
                    print('Please check LTP data format')
                isSlpData = False
            else:
                ds.append(list(map(float, line.split())))
        else:
            dln = line.split('=')
            if len(dln) == 2:
                if dln[0].strip() == 'Slopes_Xmin':
                    isSlpData = True
                else:
                    d[dln[0].strip()] = _val_interp(dln[1].strip().strip('"'))
    f.close()
    return d


def _val_interp(string):
    '''Return variable type from input string'''
    string = string.strip()
    if len(string) < 1:  # empty string
        return None
    array = string.split()
    try:
        dtype = 'float'
        if np.all(np.char.isdigit(string)):
            dtype = 'int'
        array = np.asarray(array, dtype=dtype)
    except ValueError:
        return string
    if len(array) > 1:
        return array
    return array[0]
