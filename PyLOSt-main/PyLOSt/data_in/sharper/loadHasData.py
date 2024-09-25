# coding=utf-8
'''
Created on Nov 10, 2017

Method to load sharper measurements and convert to h5
input:
    in_folder:    folder containing .has files
output:
    Single HDF5 file containing
        a. header meta data (exp conditions, instrument params ...)
        b. rawdata (slopes X/Y, height), all patches, all scans
        c. stitching versions (a new version added when stitched by new algorithm or different parameters)

@author: ADAPA
'''

import os
from fnmatch import fnmatch
from PyLOSt.data_in.sharper.has_to_h5 import HasToH5
import re


## Load measurement data folder
def loadHasData(in_folder, insId, insLoc, outFilePath, mName='MeasurementEntry', progress_bar=None):
    """
    Checks the data folder for files in the format [data_{forward/backward/static}_{scan_num}_index_{subap_num}.has]. Loops over the number of scans and calls HasToH5.hasToH5 for each scan

    :param in_folder: Raw data location
    :param insId: Instrument id
    :param insLoc: Instrument physical location
    :param outFilePath: Output h5 path
    :param mName: Measurement entry name
    :return: number of subapertures, number of fw/bw/static scans
    """
    try:
        # count number of FW/BW/Static scans
        cntFS = 0
        cntBS = 0
        cntSS = 0
        # count number patches (using first scan)
        cntPatF = 0
        cntPatB = 0
        cntPatS = 0
        cntPat = 0

        # prefix: TODO
        prefix = 'data'

        c = HasToH5(outFilePath, otype='a', mName=mName, isNewData=True)
        for f in os.listdir(in_folder):
            if fnmatch(f, '*_index_*.has'):
                fname = os.path.splitext(f)[0]
                prefix = re.split('_', fname)[0]
                if fnmatch(f, '*forward_*_index_0.has'):
                    cntFS += 1
                    s_type = 'F'
                if fnmatch(f, '*backward_*_index_0.has'):
                    cntBS += 1
                if fnmatch(f, '*static_*_index_0.has'):
                    cntSS += 1
                if fnmatch(f, '*forward_1_index_*.has'):
                    cntPatF += 1
                if fnmatch(f, '*backward_1_index_*.has'):
                    cntPatB += 1
                if fnmatch(f, '*static_1_index_*.has'):
                    cntPatS += 1
            # load reference if it exists in the selected folder
            if fnmatch(f, '*_ref_step1.has'):
                c.addReference(os.path.join(in_folder, f), loc='Data/Ref1')
        cntPat = max([cntPatF, cntPatB, cntPatS])
        cntTS = cntFS + cntBS + cntSS

        for i in range(1, cntFS + 1):
            c.hasToH5(in_folder, fname=prefix + '_forward_' + str(i), patch_count=cntPatF, h5scan='Scan_f' + str(i),
                      dircn='F', scanNo=i)
            if progress_bar:
                progress_bar.setValue(100 * i / cntTS)
        for i in range(1, cntBS + 1):
            c.hasToH5(in_folder, fname=prefix + '_backward_' + str(i), patch_count=cntPatB, h5scan='Scan_b' + str(i),
                      dircn='B', scanNo=i)
            if progress_bar:
                progress_bar.setValue(100 * (i + cntFS) / cntTS)
        for i in range(1, cntSS + 1):
            c.hasToH5(in_folder, fname=prefix + '_static_' + str(i), patch_count=cntPatS, h5scan='Scan_s' + str(i),
                      dircn='S', scanNo=i)
            if progress_bar:
                progress_bar.setValue(100 * (i + cntFS + cntBS) / cntTS)

        retData = [cntPat, cntFS, cntBS, cntSS]

        c.updateMetaData(cntArr=retData, instr_id=insId, instr_location=insLoc)
        c.finish()

        return retData
    except Exception as e:
        print('loadHasData')
        print(e)
