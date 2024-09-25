# coding=utf-8
'''
Created on Apr 9, 2018

Simple averaging of subapertures data (with piston removed for height data)

@author: ADAPA
'''
import numpy as np

from PyLOSt.algorithms.stitching.algorithm import Algorithm


class SimpleAverage(Algorithm):
    name = 'simple_avg'
    description = 'Simple average with only piston corrected for heights.'

    # Algorithm inputs
    cor_piston = False
    data_type = {'value': '', 'description': 'Input data type', 'all_values': ('', 'slope', 'height'), 'disp_type': 'S',
                 'unit': None}

    def __init__(self, stitch_options, data_in=None):
        """
        :param data_in: Data object containg raw data and other data needed for stitching
        :param stitch_options: Parameters for stitching, stitching algorithm
        """
        algo_options = stitch_options['algorithm_options']
        Algorithm.__init__(self, algo_options, data_in)
        self.data_out['creator'] = u'simple_average.py'

    def stitch_scan_item(self, key, scan_item, intensity, res_item, res_intensity, mX, mY, pix_size, prog_block=0):
        err_val = 0
        szItem = scan_item.shape
        correctors = [[0]] * len(mX)
        scan_item_cor = np.full(scan_item.shape, np.nan, dtype=scan_item.dtype)
        new_block = prog_block * 1 / len(mX)
        for j, ox in enumerate(mX):
            scan_item_cor[j] = scan_item[j]
            oy = mY[j]
            slc = (slice(oy, oy + szItem[-2]), slice(ox, ox + szItem[-1]))
            if key == 'height':
                correctors[j] = [- np.nanmean(scan_item_cor[j].flatten()) if self.cor_piston else 0.0]
                scan_item_cor[j] = scan_item_cor[j] + correctors[j]
            res_item[slc] = np.nansum([res_item[slc], scan_item_cor[j]], axis=0)
            res_intensity[slc] = np.nansum([res_intensity[slc], intensity[j]], axis=0)
            self.increment_progress(new_block)

        res_item = np.divide(res_item, res_intensity)
        err_val = self.get_algorithm_error(mX, mY, scan_item_cor, res_item)
        return err_val, res_item, correctors, scan_item_cor
