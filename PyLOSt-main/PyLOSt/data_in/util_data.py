# coding=utf-8
import os

from PyLOSt.databases.gs_table_classes import Instruments
from PyLOSt.util.commons import alertMsg


def initToH5(insId, in_folder, pf, nbScans, nbPatches, moreOpt, seperator='', directions={'': ''}, order=[],
             zfill_scans=0, zfill_patches=0):
    extSel = ''
    extension = Instruments.selectBy(instrId=insId)[0].dataFormats.split(',')
    patches = list(range(1, nbPatches + 1))
    scans = list(range(1, nbScans + 1))
    if any(moreOpt) and len(moreOpt) >= 6:
        order = moreOpt[5] if any(moreOpt[5]) else order
        directions = moreOpt[4] if any(moreOpt[4]) else directions
        if moreOpt[2] != '':
            seperator = moreOpt[2]
        if moreOpt[1] != '':
            patches = parseQMoreInt(moreOpt[1])
        if moreOpt[0] != '':
            scans = parseQMoreInt(moreOpt[0])
        if moreOpt[3] != '':
            extension = [moreOpt[3]]

    for e in extension:
        st = '' if nbScans == 0 else str(scans[0]).zfill(zfill_scans) if zfill_scans > 0 else str(scans[0])
        pt = '' if nbPatches == 0 else str(patches[0]).zfill(zfill_patches) if zfill_patches > 0 else str(patches[0])
        f_test = os.path.join(in_folder, joinFileNameSequence(pf, st, pt,
                                                              [list(directions.values())[0], seperator, e.strip(),
                                                               order]))
        if os.path.exists(f_test):
            extSel = e.strip()
            break
    if extSel == '':
        alertMsg('No data',
                 'No valid files found with selected extension(s) : ' + str(extension) + ' or with filenames ' + f_test)

    return extSel, seperator, patches, scans, directions, order


def joinFileNameSequence(prefix, scanNo, patchNo, options):
    fname = ''
    if not any(options[3]):
        options[3] = ['1', '2', '3', '4', '5']
    for i in options[3]:
        if i == '1':
            fname = fname + prefix
        if i == '2':
            fname = fname + options[0]
        if i == '3':
            fname = fname + str(scanNo)
        if i == '4':
            fname = fname + options[1]
        if i == '5':
            fname = fname + str(patchNo)
    fname = fname + '.' + options[2]
    return fname


def parseQMoreInt(quser):
    try:
        retArr = []
        if quser == '':
            return retArr
        else:
            A = quser.split(',')
            for a in A:
                if '-' in a:
                    an = a.split('-')
                    for i in range(int(an[0]), int(an[1]) + 1):
                        retArr.append(i)
                else:
                    retArr.append(int(a))
            return retArr
    except Exception as e:
        print('parseQMore <- util_functions')
        print(e)
