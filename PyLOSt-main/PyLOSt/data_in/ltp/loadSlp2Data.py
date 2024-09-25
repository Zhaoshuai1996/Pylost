# coding=utf-8
'''
Created on Aug 6, 2018

@author: ADAPA
'''
from PyLOSt.data_in.ltp.slp2_to_h5 import Slp2ToH5

## Load measurement data folder
from PyLOSt.data_in.util_data import initToH5


def loadSlp2Data(in_folder, insId, insLoc, outFilePath, pf='JTEC-Flat-st4-R0-P', nbScans=1, nbPatches=0,
                 mName='MeasurementEntry', moreOpt=[], progress_bar=None):
    """
    Loops over the number of scans and calls Slp2ToH5.slpToH5 for each scan

    :param in_folder: Raw data location
    :param insId: Instrument id
    :param insLoc: Instrument physical location
    :param outFilePath: Output h5 path
    :param pf: File name prefix
    :param nbScans: Number of scans
    :param nbPatches: Number of subapertures
    :param mName: Measurement entry name
    :return: Number of scans, number of subapertures
    """
    try:
        extSel, seperator, patches, scans, directions, order = initToH5(insId, in_folder, pf, nbScans, nbPatches,
                                                                        moreOpt,
                                                                        seperator='_scan-' if nbPatches > 0 else '',
                                                                        directions={'F': 'FWD_', 'B': 'BWD_'},
                                                                        order=['1', '5', '4', '2', '3'], zfill_scans=2,
                                                                        zfill_patches=2)
        c = Slp2ToH5(outFilePath, otype='a', mName=mName, isNewData=True)
        idx = 0
        for i in scans:
            for d in directions.keys():
                idx = idx + 1
                scanGrpName = 'Scan_' + str(i) if len(directions.keys()) == 1 else 'Scan_' + d.lower() + str(i)
                c.slpToH5(in_folder, prefix=pf, nbPatches=nbPatches, h5scan=scanGrpName, dircn=d, scanNo=i, scanIdx=idx,
                          nbSelScans=len(scans),
                          patches_selected=patches, options=[directions[d], seperator, extSel, order],
                          progress_bar=progress_bar)

        retData = [nbScans, nbPatches]

        c.updateMetaData(cntArr=retData, instr_id=insId, instr_location=insLoc)
        c.finish()

        return retData
    except Exception as e:
        print('loadSlp2Data')
        print(e)
