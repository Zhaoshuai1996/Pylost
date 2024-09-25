# coding=utf-8
'''
Created on Aug 6, 2018

@author: adapa
'''
from PyLOSt.data_in.fizeau.dat_to_h5 import DatToH5

## Load measurement data folder
from PyLOSt.data_in.fizeau.datx.datx_to_h5 import DatxToH5
from PyLOSt.data_in.util_data import initToH5
from PyLOSt.util.commons import alertMsg


def loadFizData(in_folder, insId, insLoc, outFilePath, pf='', nbScans=1, nbPatches=32, mName='MeasurementEntry',
                moreOpt=[], progress_bar=None):
    """
    Loops over the number of scans and calls DatToH5.datToH5 for each scan

    :param in_folder: Raw data location
    :param insId: Instrument id
    :param insLoc: Instrument physical location
    :param outFilePath: Output h5 path
    :param pf: File name prefix
    :param nbScans: Number of scans
    :param nbPatches: Number of subapertures
    :param mName: Measurement entry name
    :param dircn: Forward or backward
    :return: Number of subapertures, number of scans
    """
    try:
        extSel, seperator, patches, scans, directions, order = initToH5(insId, in_folder, pf, nbScans, nbPatches,
                                                                        moreOpt, seperator='-P')

        if extSel == 'dat':
            c = DatToH5(outFilePath, otype='a', mName=mName, isNewData=True)
        elif extSel == 'datx':
            c = DatxToH5(outFilePath, otype='a', mName=mName, isNewData=True)
        else:
            alertMsg('No data', 'No *.dat or *.datx files found')
            return 0

        idx = 0
        for i in scans:
            for d in directions.keys():
                idx = idx + 1
                scanGrpName = 'Scan_' + str(i) if len(directions.keys()) == 1 else 'Scan_' + d.lower() + str(i)
                c.toH5(in_folder, prefix=pf, patch_count=nbPatches, h5scan=scanGrpName, dircn=d, scanNo=i, scanIdx=idx,
                       nbSelScans=len(scans),
                       patches_selected=patches, options=[directions[d], seperator, extSel, order],
                       progress_bar=progress_bar)

        retData = [nbPatches, nbScans]

        c.updateMetaData(cntArr=retData, instr_id=insId, instr_location=insLoc)
        c.finish()

        return 1
    except Exception as e:
        print('loadFizData')
        print(e)
